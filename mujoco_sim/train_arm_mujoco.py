# -*- coding: utf-8 -*-
"""ARM_BCRL006 -- the single simulator entry point.

One file, one environment, six modes -- the same structure
``mujoco_sim/train_arm_mujoco.py`` had in ARM_BCRL005, so the pipeline scripts
in ``src/`` drive this the way they drove that:

    --mode collect     phase 1    scripted IK expert -> demonstrations
    --mode mimicgen    phase 1.5  warp them onto new scenes, keep successes
    --mode eval_only   phase 4/9  closed-loop evaluation of a baseline or ONNX
    --mode rl          phase 6    PPO fine-tuning from the BC weights
    --mode distill     phase 7    RL champion re-collects camera demos
    --mode play        --         watch it, or record the report video

Everything is CPU-side MuJoCo; only the learning phases touch the GPU.

SHARDING (``--shard i --num_shards N``): the camera-bearing modes are limited
by the graphics stack, not by physics, and rendering is embarrassingly
parallel. ``src/collect_data.py`` and ``src/distill_collect.py`` launch N of
these processes -- each with its own model, data and GL context -- and merge
the shards. A shard offsets the seed, so the union is still one i.i.d. sample.

Usage:
    python mujoco_sim/train_arm_mujoco.py --mode collect --num_envs 32 \
        --set collect.episodes=8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf  # noqa: E402

parser = argparse.ArgumentParser(description="ARM_BCRL006 MuJoCo pick&place")
parser.add_argument("--mode", default="rl",
                    choices=["collect", "mimicgen", "eval_only", "rl", "distill", "play"])
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--baseline", default="checkpoint",
                    choices=["random", "scripted_expert", "checkpoint"])
parser.add_argument("--onnx_path", default=None)
parser.add_argument("--eval_out", default=None, help="eval report basename override")
parser.add_argument("--eval_dr", default=None, choices=["on", "off"],
                    help="force domain randomisation on/off for THIS run "
                         "(src/ablation.py uses it to hold the test env fixed)")
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--out_name", default=None, help="collect/distill: output .pt name")
parser.add_argument("--teacher_onnx", default=None)
parser.add_argument("--video", default=None, help="play: write an mp4 here instead of a window")
parser.add_argument("--video_seconds", type=float, default=60.0)
parser.add_argument("--config_dir", default=os.path.join(REPO_ROOT, "configs"))
parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                    help="OmegaConf dotlist overrides, e.g. --set rl.lr=0.0001")
args_cli = parser.parse_args()


def _load_cfg() -> "OmegaConf":
    d = args_cli.config_dir
    base = OmegaConf.create({
        "env": OmegaConf.load(os.path.join(d, "env", "mycobot_pickplace.yaml")),
        "bc": OmegaConf.load(os.path.join(d, "bc", "robomimic_bc.yaml")),
        "rl": OmegaConf.load(os.path.join(d, "rl", "ppo.yaml")),
        "visuomotor": OmegaConf.load(os.path.join(d, "visuomotor", "act.yaml")),
        "mimicgen": OmegaConf.load(os.path.join(d, "mimicgen", "default.yaml")),
        "mlops": OmegaConf.load(os.path.join(d, "mlops", "default.yaml")),
        "experiment_name": "arm_bcrl006_pickplace",
        "seed": 42,
        "device": "cuda",
        "collect": {"num_envs": 32, "episodes": 8, "action_noise": 0.1,
                    "novice_ratio": 0.1, "keep_failed": False, "camera": True},
        "eval": {"num_envs": 64, "episodes": 4, "max_steps": 300},
    })
    cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(list(args_cli.overrides)))
    OmegaConf.resolve(cfg)
    if args_cli.seed is not None:
        cfg.seed = int(args_cli.seed)
    if args_cli.eval_dr is not None:
        cfg.env.domain_rand.enable = (args_cli.eval_dr == "on")
    return cfg


_cfg = _load_cfg()

from mujoco_sim.env import MuJoCoPickPlaceEnv, ACT_DIM  # noqa: E402
from mujoco_sim.policies import ExpertController, OnnxPolicy  # noqa: E402

# Wrist camera: recorded during collect / mimicgen / distill, shown during
# play, and needed when evaluating an IMAGE policy. Everything else is
# render-free, which is what keeps the state-only phases fast -- rendering,
# not physics, is this pipeline's bottleneck.
def _needs_camera() -> bool:
    if args_cli.mode in ("distill", "play"):
        return True                 # distillation EXISTS to produce frames
    if args_cli.mode in ("collect", "mimicgen"):
        # collect.camera=false skips rendering entirely. The image-quality gate
        # and the visuomotor stage need frames, so the full pipeline leaves it
        # on -- but the 2x2 study measures a STATE policy, and rendering there
        # would cost 20x the wall clock for pixels nothing reads.
        return bool(_cfg.collect.get("camera", True))
    if args_cli.mode == "eval_only" and args_cli.baseline == "checkpoint" \
            and args_cli.onnx_path:
        import onnxruntime as _ort
        names = [i.name for i in _ort.InferenceSession(
            args_cli.onnx_path, providers=["CPUExecutionProvider"]).get_inputs()]
        # ask the GRAPH, not the file name: an ONNX that declares an "rgb"
        # input needs frames, and nothing else does
        return "rgb" in names
    return False


CAMERA_ON = _needs_camera()
DATA_DIR = os.path.join(REPO_ROOT, "artifacts", "data")
EVAL_DIR = os.path.join(REPO_ROOT, "artifacts", "eval")
DEPLOY_DIR = os.path.join(REPO_ROOT, "artifacts", "deploy")


def _shard_seed() -> int:
    return int(_cfg.seed) + 10007 * int(args_cli.shard)


def _make_env(num_envs: int | None = None, camera: bool | None = None):
    return MuJoCoPickPlaceEnv(
        _cfg,
        num_envs=int(num_envs if num_envs is not None else args_cli.num_envs),
        with_camera=CAMERA_ON if camera is None else bool(camera),
        seed=_shard_seed())


# ---------------------------------------------------------------------------
# closed-loop evaluation (one loop; stateful policies get their reset hook
# called on episode boundaries)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_policy(env, act_fn, episodes: int, max_steps: int, reset_hook=None,
                    seed: int | None = None) -> dict:
    """Run the policy in closed loop and report what a reviewer needs to know.

    Metrics: success_rate, time_to_success (successful episodes only),
    mean_max_lift (peak block height -- partial progress, so a policy that
    grasps but never places is distinguishable from one that never touches the
    block), mean_place_err, mean_reward, and the safety telemetry
    (mean/worst min_tool_z: how close the tool actually got to the bench).
    """
    n = env.num_envs
    # ``seed`` makes the battery REPRODUCIBLE: the checkpoint sweep passes a
    # fixed one so every candidate is judged on the same scenes (a paired
    # comparison -- the variance between checkpoints is then the difference
    # between policies, not between episode draws), and the final measurement
    # passes a different one so it is an INDEPENDENT re-estimate of the winner
    # rather than the number the winner was selected by.
    obs = env.reset(seed=seed)[0]["policy"]
    if reset_hook is not None:
        reset_hook.reset_idx(torch.arange(n))
    steps = torch.zeros(n)
    ret = torch.zeros(n)
    recs = {k: [] for k in ("success", "steps", "lift", "place_err", "reward", "min_tool_z")}

    total_target, completed, step = episodes * n, 0, 0
    while completed < total_target and step < max_steps:
        obs, rew, term, trunc, _ = env.step(act_fn(obs))
        obs = obs["policy"]
        step += 1
        steps += 1
        ret += rew.cpu()
        done_ids = (term | trunc).nonzero(as_tuple=False).flatten()
        for i in done_ids.tolist():
            # `term` from env.step() itself, never a fresh is_success(): the
            # auto-reset already replaced the terminal state
            recs["success"].append(float(bool(term[i])))
            recs["steps"].append(float(steps[i]))
            recs["lift"].append(float(env.final_max_obj_z[i]))
            recs["place_err"].append(float(env.final_place_err[i]))
            recs["reward"].append(float(ret[i]))
            recs["min_tool_z"].append(float(env.final_min_tool_z[i]))
            completed += 1
            steps[i] = 0
            ret[i] = 0
        if len(done_ids) > 0 and reset_hook is not None:
            reset_hook.reset_idx(done_ids)

    def _arr(k):
        return np.array(recs[k]) if recs[k] else np.zeros(1)

    succ, stp = _arr("success"), _arr("steps")
    perr = _arr("place_err")
    perr = perr[np.isfinite(perr)]
    tz = _arr("min_tool_z")
    tz = tz[np.isfinite(tz)]
    return {
        "episodes": int(len(recs["success"])),
        "success_rate": float(succ.mean()),
        "time_to_success": float(stp[succ > 0.5].mean()) if (succ > 0.5).any() else float("nan"),
        "mean_max_lift": float(_arr("lift").mean()),
        "mean_place_err": float(perr.mean()) if len(perr) else float("nan"),
        "mean_reward": float(_arr("reward").mean()),
        # mean_min_tool_z well above env.floor.clearance means the policy is
        # NOT relying on the guard; a value pinned at the plane means it is.
        "mean_min_tool_z": float(tz.mean()) if len(tz) else float("nan"),
        "worst_min_tool_z": float(tz.min()) if len(tz) else float("nan"),
    }


# ---------------------------------------------------------------------------
# collect (phase 1) & distill (phase 7) share one recording loop
# ---------------------------------------------------------------------------
def _record_rollouts(env, act_fn, reset_hook, episodes_target: int, noise: float,
                     novice_ratio: float, keep_failed: bool, out_name: str,
                     summary_name: str) -> dict:
    n = env.num_envs
    obs = env.reset()[0]["policy"]
    if reset_hook is not None:
        reset_hook.reset_idx(torch.arange(n))
    is_novice = torch.rand(n) < novice_ratio

    cur_obs = [[] for _ in range(n)]
    cur_act = [[] for _ in range(n)]
    cur_rgb = [[] for _ in range(n)]
    kept_o, kept_a, kept_rgb = [], [], []
    n_traj = n_success = n_kept = 0
    max_steps = (episodes_target // max(1, n) + 1) * int(_cfg.env.episode_len_steps) + 5

    step = 0
    while n_traj < episodes_target and step < max_steps:
        clean = act_fn(obs)
        # DART-style: execute noisy, label clean. The policy then sees states
        # a perfect demonstrator never visits, together with the action that
        # recovers from them -- the single cheapest fix for BC's compounding
        # error, and it costs one extra line.
        exec_a = torch.clamp(clean + torch.randn_like(clean) * noise, -1, 1) \
            if noise > 0 else clean
        if novice_ratio > 0:
            novice = torch.rand_like(clean) * 2 - 1
            label = torch.where(is_novice.unsqueeze(1), novice, clean)
            exec_a = torch.where(is_novice.unsqueeze(1), novice, exec_a)
        else:
            label = clean

        rgb_frame = env.camera_rgb_uint8() if env.with_camera else None
        for i in range(n):
            cur_obs[i].append(obs[i].detach().cpu())
            cur_act[i].append(label[i].detach().cpu())
            if rgb_frame is not None:
                cur_rgb[i].append(rgb_frame[i])

        obs, _, term, trunc, _ = env.step(exec_a)
        obs = obs["policy"]
        done_ids = (term | trunc).nonzero(as_tuple=False).flatten()
        for i in done_ids.tolist():
            succ = bool(term[i])
            if succ or keep_failed:
                kept_o.append(torch.stack(cur_obs[i]))
                kept_a.append(torch.stack(cur_act[i]))
                if env.with_camera:
                    kept_rgb.append(torch.stack(cur_rgb[i]))
                n_kept += 1
            n_traj += 1
            n_success += int(succ)
            cur_obs[i], cur_act[i], cur_rgb[i] = [], [], []
        if len(done_ids) > 0 and reset_hook is not None:
            reset_hook.reset_idx(done_ids)
        step += 1
        if step % 50 == 0:
            print(f"  [shard {args_cli.shard}] step {step}/{max_steps}, "
                  f"trajectories {n_traj}/{episodes_target}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    demos = {"obs": kept_o, "act": kept_a}
    if env.with_camera:
        demos["rgb"] = kept_rgb
    final = os.path.join(DATA_DIR, out_name)
    tmp = final + ".tmp"
    torch.save(demos, tmp)          # atomic: never leave a half-written archive
    os.replace(tmp, final)

    summary = {
        "trajectories": n_traj,
        "source_success_rate": n_success / max(1, n_traj),
        "kept_trajectories": n_kept,
        "dropped_trajectories": n_traj - n_kept,
        "kept_transitions": int(sum(o.shape[0] for o in kept_o)),
    }
    with open(os.path.join(DATA_DIR, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _shard_names(base: str) -> tuple[str, str]:
    """collect/distill write one file per shard; src/ merges them."""
    stem = base[:-3] if base.endswith(".pt") else base
    if int(args_cli.num_shards) <= 1:
        return f"{stem}.pt", f"{stem}_summary.json"
    return (f"{stem}.shard{args_cli.shard}.pt",
            f"{stem}_summary.shard{args_cli.shard}.json")


def run_collect() -> None:
    env = _make_env(int(_cfg.collect.num_envs), bool(_cfg.collect.get("camera", True)))
    expert = ExpertController(env)
    out_name, summary_name = _shard_names(args_cli.out_name or "bc_demos.pt")
    episodes_target = int(_cfg.collect.episodes) * env.num_envs
    print(f"[collect] shard {args_cli.shard}/{args_cli.num_shards}: "
          f"~{episodes_target} expert pick&place trajectories", flush=True)
    summary = _record_rollouts(
        env, lambda o: expert.act(), expert, episodes_target,
        noise=float(_cfg.collect.action_noise),
        novice_ratio=float(_cfg.collect.novice_ratio),
        keep_failed=bool(_cfg.collect.keep_failed),
        out_name=out_name, summary_name=summary_name)
    print("[collect]", json.dumps(summary, indent=2))


def _find_teacher_onnx() -> str | None:
    if args_cli.teacher_onnx:
        return args_cli.teacher_onnx
    for name in ("arm_bcrl006_rl_policy.onnx", "arm_bcrl006_bc_policy.onnx"):
        p = os.path.join(DEPLOY_DIR, name)
        if os.path.exists(p):
            return p
    return None


def run_distill() -> None:
    """Teacher-student data generation: the RL champion (a STATE policy that
    reads object poses -- information only the simulator has) demonstrates with
    the wrist camera on; the recorded (rgb, qpos) -> action pairs train the
    deployable visuomotor policy, which needs no object pose at runtime."""
    d = _cfg.visuomotor.distill
    env = _make_env(int(d.num_envs), True)
    teacher_path = _find_teacher_onnx()
    if teacher_path is not None:
        print(f"[distill] teacher: {teacher_path}")
        teacher = OnnxPolicy(teacher_path, env)
        act_fn, reset_hook = teacher, teacher
    else:
        print("[distill] no deployed state policy found -- falling back to the "
              "scripted expert")
        expert = ExpertController(env)
        act_fn, reset_hook = (lambda o: expert.act()), expert

    out_name, summary_name = _shard_names(args_cli.out_name or "visuomotor_demos.pt")
    episodes_target = int(d.episodes) * env.num_envs
    print(f"[distill] shard {args_cli.shard}/{args_cli.num_shards}: "
          f"~{episodes_target} teacher trajectories (camera on)", flush=True)
    summary = _record_rollouts(
        env, act_fn, reset_hook, episodes_target, noise=0.05, novice_ratio=0.0,
        keep_failed=not bool(d.min_success),
        out_name=out_name, summary_name=summary_name)
    summary["teacher"] = teacher_path or "scripted_expert"
    with open(os.path.join(DATA_DIR, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[distill]", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# mimicgen (phase 1.5)
# ---------------------------------------------------------------------------
def run_mimicgen() -> None:
    from mimicgen.generation import DataGenInfoPool, TrajectoryGenerator
    from mimicgen.pose_np import make_pose

    mg = _cfg.mimicgen
    ann = os.path.join(DATA_DIR, str(mg.annotated_file))
    if not os.path.exists(ann):
        raise FileNotFoundError(
            f"{ann} not found -- src/mimicgen_generate.py runs mimicgen/annotate.py first")
    pool = DataGenInfoPool.load(ann, max_demos=int(mg.max_source_demos))
    gen = TrajectoryGenerator(pool, selection=str(mg.selection), nn_k=int(mg.nn_k),
                              interpolation_steps=int(mg.interpolation_steps),
                              action_noise=float(mg.action_noise), seed=_shard_seed())
    print(f"[mimicgen] shard {args_cli.shard}: pool of {len(pool.episodes)} annotated "
          f"demos, {pool.num_subtasks} subtasks, selection={mg.selection}", flush=True)

    # Camera follows mimicgen.camera (default OFF, forwarded by
    # src/mimicgen_generate.py as collect.camera). Generated demos feed the
    # STATE policy; rendering them would add gigabytes for pixels nothing
    # reads. Hardcoding True here cost one run of the full pipeline.
    env = _make_env(int(mg.num_envs), CAMERA_ON)
    kin = env.tool_kin
    from arm_common.ik import DLSPositionIK
    ik = DLSPositionIK(kin, damping=float(_cfg.env.expert.get("ik_damping", 0.05)))
    attempts_target = int(mg.episodes) * env.num_envs

    kept_o, kept_a, kept_rgb = [], [], []
    n_attempt = n_success = 0
    env.reset()

    while n_attempt < attempts_target:
        # one generation attempt per environment, executed in parallel: plan
        # the whole episode for every env, then step them together.
        plans = []
        for i in range(env.num_envs):
            obj = env.obj_pos_env()[i].numpy().astype(np.float64)
            tgt = env.place_target[i].numpy().astype(np.float64)
            eef = env.eef_pos_env()[i].numpy().astype(np.float64)
            T_eef = make_pose(eef)
            seq_pos, seq_grip = [], []
            for s, ref_pos in enumerate((obj, tgt)):
                poses, grip, _src = gen.plan_subtask(s, make_pose(ref_pos), T_eef)
                seq_pos.append(poses[:, :3, 3])
                seq_grip.append(grip)
                T_eef = poses[-1]
            plans.append((np.concatenate(seq_pos), np.concatenate(seq_grip)))
        horizon = min(max(len(p[0]) for p in plans), int(_cfg.env.episode_len_steps))

        cur_o = [[] for _ in range(env.num_envs)]
        cur_a = [[] for _ in range(env.num_envs)]
        cur_r = [[] for _ in range(env.num_envs)]
        alive = np.ones(env.num_envs, dtype=bool)
        obs = env.get_observations()["policy"]
        for t in range(horizon):
            tgt_pos = np.stack([p[0][min(t, len(p[0]) - 1)] for p in plans])
            grip = np.array([p[1][min(t, len(p[1]) - 1)] for p in plans], dtype=np.float32)
            q = env.arm_q()
            q_des = ik.solve(q, torch.as_tensor(tgt_pos, dtype=torch.float32))
            arm_a = ((q_des - q) / env.action_scale).clamp(-1.0, 1.0)
            grip_a = torch.as_tensor(np.where(grip > 0, 1.0, -1.0),
                                     dtype=torch.float32).unsqueeze(1)
            action = torch.cat([arm_a, grip_a], dim=-1)
            rgb = env.camera_rgb_uint8() if env.with_camera else None
            for i in range(env.num_envs):
                if alive[i]:
                    cur_o[i].append(obs[i].cpu())
                    cur_a[i].append(action[i].cpu())
                    if rgb is not None:
                        cur_r[i].append(rgb[i])
            obs, _, term, trunc, _ = env.step(action)
            obs = obs["policy"]
            for i in (term | trunc).nonzero(as_tuple=False).flatten().tolist():
                if not alive[i]:
                    continue
                alive[i] = False
                n_attempt += 1
                if bool(term[i]):               # the env's OWN success term
                    n_success += 1
                    kept_o.append(torch.stack(cur_o[i]))
                    kept_a.append(torch.stack(cur_a[i]))
                    if env.with_camera:
                        kept_rgb.append(torch.stack(cur_r[i]))
            if not alive.any():
                break
        for i in range(env.num_envs):           # unfinished attempts count as failures
            if alive[i]:
                n_attempt += 1
        env.reset()
        print(f"  [mimicgen shard {args_cli.shard}] attempts {n_attempt}/{attempts_target}, "
              f"kept {n_success} (yield {n_success / max(1, n_attempt):.2f})", flush=True)

    out_name, summary_name = _shard_names("bc_demos_mimicgen_raw.pt")
    os.makedirs(DATA_DIR, exist_ok=True)
    final = os.path.join(DATA_DIR, out_name)
    out = {"obs": kept_o, "act": kept_a}
    if env.with_camera:
        out["rgb"] = kept_rgb
    torch.save(out, final + ".tmp")
    os.replace(final + ".tmp", final)
    summary = {"attempts": n_attempt, "generated": n_success,
               "yield": n_success / max(1, n_attempt),
               "source_demos": len(pool.episodes),
               "generated_transitions": int(sum(o.shape[0] for o in kept_o))}
    with open(os.path.join(DATA_DIR, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[mimicgen]", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
def run_eval_only() -> None:
    env = _make_env(int(_cfg.eval.num_envs))
    baseline = args_cli.baseline
    reset_hook = None
    if baseline == "random":
        def act_fn(obs):
            return torch.rand(env.num_envs, ACT_DIM) * 2 - 1
    elif baseline == "scripted_expert":
        expert = ExpertController(env)
        reset_hook = expert
        def act_fn(obs):                                        # noqa: E306
            return expert.act()
    else:
        onnx_path = args_cli.onnx_path or os.path.join(
            DEPLOY_DIR, "arm_bcrl006_bc_policy.onnx")
        policy = OnnxPolicy(onnx_path, env,
                            execute_horizon=int(_cfg.visuomotor.execute_horizon))
        reset_hook = policy
        act_fn = policy
        print(f"[eval] onnx: {onnx_path} type={policy.policy_type} chunk={policy.chunk}")

    metrics = evaluate_policy(env, act_fn, int(_cfg.eval.episodes),
                              int(_cfg.eval.max_steps), reset_hook)
    metrics["domain_rand"] = bool(_cfg.env.domain_rand.enable)
    os.makedirs(EVAL_DIR, exist_ok=True)
    tag = args_cli.eval_out or baseline
    with open(os.path.join(EVAL_DIR, f"eval_report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[eval] {tag}: {json.dumps(metrics)}")


def run_play() -> None:
    from mujoco_sim.viewer import play
    play(_cfg, args_cli)


def run_rl() -> None:
    from mujoco_sim.rl_ppo import train_ppo
    train_ppo(_cfg, args_cli, _make_env, evaluate_policy)


def main() -> None:
    torch.manual_seed(_shard_seed())
    np.random.seed(_shard_seed())
    {"collect": run_collect, "mimicgen": run_mimicgen, "eval_only": run_eval_only,
     "distill": run_distill, "play": run_play, "rl": run_rl}[args_cli.mode]()


if __name__ == "__main__":
    main()
