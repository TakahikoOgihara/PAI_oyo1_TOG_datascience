# -*- coding: utf-8 -*-
"""Phase 6 -- PPO fine-tuning FROM the BC policy (stable-baselines3).

WHY THIS PHASE EXISTS. BC is capped by its demonstrator: the scripted expert is
the ceiling, and a cloned policy is strictly worse than what it cloned. PPO
from scratch, on the other hand, burns hours rediscovering what the
demonstrations already knew. Initialising PPO from the BC weights takes both --
and the delta is MEASURED (src/rl_finetune.py compares the two eval reports),
not asserted. ``rl.bc_init=false`` quantifies exactly what the initialisation
buys.

WHY THE FOUR STABILISERS. ARM_BCRL005's first real run watched a BC policy at
success 0.606 collapse to 0.000 over 600 iterations. It was not reward hacking
and not entropy collapse: the CRITIC diverged. A randomly initialised value head
facing BC-policy returns of 800-1900 produced a value loss of 4e4, so
``advantage = returns - values`` was noise and PPO faithfully followed it. Once
the policy stops succeeding, the sparse success gradient is gone and it never
comes back. The fixes, in order of effect:

  1. RETURN NORMALISATION -- put the critic's regression target at O(1).
     ``VecNormalize(norm_reward=True)`` is the library's own implementation of
     exactly what ARM_BCRL005 hand-rolled.
  2. CRITIC WARM-UP -- freeze the actor for the first N iterations so the value
     function catches up before anything is allowed to change the policy.
  3. A SMALLER LEARNING RATE -- 3e-4 is a from-scratch number.
  4. BC ANCHOR -- an auxiliary gradient step pulling the actor back toward the
     BC snapshot, on the states the rollout actually visited, decayed to zero so
     RL can still surpass BC. Same idea as the KL-to-SFT penalty in RLHF, and as
     DAPG's BC gradient term (Rajeswaran et al. 2018).

The decay is applied to the anchor optimiser's LEARNING RATE, not to the loss
coefficient. That distinction cost ARM_BCRL005 a run: Adam's update
``lr * m / (sqrt(v) + eps)`` is INVARIANT to a constant scaling of the loss, so
ramping a coefficient 1.0 -> 0.0 does nothing until it is exactly zero, at which
point the constraint vanishes off a cliff. What is meant to decay is step size.

WHY STABLE-BASELINES3 AND NOT RSL-RL (ARM_BCRL005's choice): RSL-RL's runner is
written against Isaac Lab's vectorised-env wrapper and is not usable outside it.
SB3 is the mainstream general-purpose RL library, its ``VecEnv`` contract is
stable and documented, it writes TensorBoard scalars natively, and the algorithm
is the same clipped-objective PPO.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mlops import export as onnx_export  # noqa: E402
from mlops import registry  # noqa: E402
from mlops.tracking import mlflow_run, set_seed, wandb_env, wandb_run  # noqa: E402

OBS_DIM, ACT_DIM = 31, 7


# ---------------------------------------------------------------------------
class MjVecEnv(VecEnv):
    """SB3 ``VecEnv`` over ``MuJoCoPickPlaceEnv``.

    The environment is ALREADY vectorised and already auto-resets, so this is a
    thin adapter rather than a wrapper stack: no SubprocVecEnv, no per-env
    Python objects, one ``mj_step`` for the whole batch.
    """

    def __init__(self, env):
        self.env = env
        super().__init__(
            env.num_envs,
            spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
            spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))
        self._actions = None

    def reset(self):
        return self.env.reset()[0]["policy"].numpy().astype(np.float32)

    def step_async(self, actions):
        self._actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)

    def step_wait(self):
        obs, rew, term, trunc, _ = self.env.step(self._actions)
        flat = obs["policy"].numpy().astype(np.float32)
        dones = (term | trunc).numpy()
        infos = [{} for _ in range(self.num_envs)]
        for i in np.nonzero(dones)[0]:
            i = int(i)
            # SB3 bootstraps the value of a TRUNCATED episode and does not
            # bootstrap a TERMINATED one. Success here is a true terminal state;
            # a timeout is not, and mislabelling the two biases every return
            # estimate near the horizon.
            infos[i]["TimeLimit.truncated"] = bool(trunc[i]) and not bool(term[i])
            infos[i]["terminal_observation"] = flat[i]
        return flat, rew.numpy().astype(np.float32), dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_images(self):
        return [None] * self.num_envs


# ---------------------------------------------------------------------------
class BCAnchor:
    """The auxiliary gradient step toward the BC snapshot (module docstring)."""

    def __init__(self, actor, reference, lr: float, coef: float, decay_iters: int,
                 minibatch: int, max_grad_norm: float, start_iter: int,
                 min_coef: float = 0.0):
        self.actor = actor
        self.reference = reference
        self.coef = float(coef)
        self.min_coef = float(min_coef)
        self.decay_iters = max(1, int(decay_iters))
        self.minibatch = int(minibatch)
        self.max_grad_norm = float(max_grad_norm)
        self.start_iter = int(start_iter)
        self.base_lr = float(lr)
        # its own optimiser: the anchor must not perturb PPO's schedule, and
        # PPO's adaptive learning rate must not perturb the anchor's decay
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=self.base_lr)

    def coef_at(self, it: int) -> float:
        if it < self.start_iter:
            return 0.0
        progress = (it - self.start_iter) / self.decay_iters
        return max(self.min_coef * self.coef, self.coef * max(0.0, 1.0 - progress))

    def step(self, obs: torch.Tensor, it: int) -> float:
        coef = self.coef_at(it)
        if coef <= 0.0 or obs is None or obs.shape[0] == 0:
            return 0.0
        for g in self.optimizer.param_groups:      # decay the STEP SIZE
            g["lr"] = self.base_lr * (coef / self.coef if self.coef > 0 else 0.0)
        idx = torch.randint(0, obs.shape[0], (min(self.minibatch, obs.shape[0]),),
                            device=obs.device)
        batch = obs[idx]
        with torch.no_grad():
            target = self.reference(batch)
        loss = nn.functional.mse_loss(self.actor(batch), target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach())


class StabilisedPPO(PPO):
    """PPO + critic warm-up + BC anchor. Everything else is stock SB3."""

    def setup_stabilisers(self, critic_warmup: int, anchor: BCAnchor | None) -> None:
        self.critic_warmup = int(critic_warmup)
        self.anchor = anchor
        self.iteration = 0
        self._actor_frozen = False

    def _set_actor_trainable(self, flag: bool) -> None:
        for m in (self.policy.mlp_extractor.policy_net, self.policy.action_net):
            for p in m.parameters():
                p.requires_grad_(flag)
        self.policy.log_std.requires_grad_(flag)
        self._actor_frozen = not flag

    def train(self) -> None:
        warm = self.iteration < self.critic_warmup
        if warm != self._actor_frozen:
            self._set_actor_trainable(not warm)
            print(f"[rl] iteration {self.iteration}: actor "
                  f"{'FROZEN (critic warm-up)' if warm else 'unfrozen'}", flush=True)
        super().train()
        if self.anchor is not None and not warm:
            # anchor on the states the rollout ACTUALLY visited: constraining
            # the policy on random observations constrains it where it never goes
            obs = torch.as_tensor(
                self.rollout_buffer.observations.reshape(-1, OBS_DIM),
                dtype=torch.float32, device=self.device)
            self.logger.record("train/bc_anchor_mse", self.anchor.step(obs, self.iteration))
            self.logger.record("train/bc_anchor_coef", self.anchor.coef_at(self.iteration))
        self.iteration += 1


class PolicyCheckpoint(BaseCallback):
    """Save the POLICY STATE DICT, not an SB3 ``.zip``.

    The actor head is modified after construction (a tanh is appended to match
    robomimic's squashed output), so ``PPO.load`` would rebuild a policy whose
    parameter NAMES no longer match the archive. A state dict loaded back into
    the same, already-modified policy object has no such problem -- and it is
    what the checkpoint sweep needs anyway.
    """

    def __init__(self, save_path: str, every_iters: int, rollout_steps: int, num_envs: int):
        super().__init__()
        self.save_path = save_path
        self.every = max(1, int(every_iters))
        self.iter_steps = int(rollout_steps) * int(num_envs)
        self._last = -1

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        it = self.num_timesteps // self.iter_steps
        if it != self._last and it % self.every == 0:
            self._last = it
            torch.save(self.model.policy.state_dict(),
                       os.path.join(self.save_path, f"model_{it:05d}.pt"))


class WandbScalars(BaseCallback):
    """Mirror SB3's logger into W&B so training is watchable LIVE, not only
    afterwards. TensorBoard receives the same scalars natively (SB3 writes them
    itself) and MLflow receives the summary at the end -- three tools, three
    jobs (configs/mlops/default.yaml)."""

    def __init__(self, wb):
        super().__init__()
        self.wb = wb

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        d = getattr(self.model.logger, "name_to_value", None)
        if not d:
            return
        self.wb.log({f"ppo/{k}": float(v) for k, v in d.items()
                     if isinstance(v, (int, float, np.floating))},
                    step=int(self.num_timesteps))


class OnnxActor(nn.Module):
    """Deterministic policy graph: obs -> action. What actually ships."""

    def __init__(self, policy):
        super().__init__()
        self.extractor = copy.deepcopy(policy.features_extractor)
        self.pi = copy.deepcopy(policy.mlp_extractor.policy_net)
        self.head = copy.deepcopy(policy.action_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pi(self.extractor(obs))).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
def _linears(module: nn.Module) -> list:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _actor_module(model) -> nn.Sequential:
    return nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)


def find_bc_checkpoint(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    runs = os.path.join(REPO_ROOT, "artifacts", "runs")
    ckpts = glob.glob(os.path.join(runs, "**", "*.pth"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(
            f"no robomimic BC checkpoint under {runs} -- run phase 3 "
            "(scripts/train_bc_robomimic.py) before phase 6, or set rl.bc_init=false")
    best = [c for c in ckpts if "_best_validation_" in c]
    return max(best or ckpts, key=os.path.getmtime)


def _bc_actor(ckpt_path: str, device):
    import robomimic.utils.file_utils as FileUtils
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device,
                                                 verbose=False)
    return policy.policy.nets["policy"]


def init_actor_from_bc(model, ckpt_path: str, device) -> int:
    """Copy the robomimic BC MLP into the PPO actor, layer by layer.

    Requires identical shapes, which ``bc.hidden == rl.hidden`` and
    ``rl.activation=relu`` guarantee; anything else is a configuration bug and
    is raised rather than silently half-copied.
    """
    src = _linears(_bc_actor(ckpt_path, device))
    dst = _linears(model.policy.mlp_extractor.policy_net) + _linears(model.policy.action_net)
    if len(src) != len(dst):
        raise RuntimeError(
            f"BC actor has {len(src)} linear layers, the PPO actor has {len(dst)}. "
            "bc.hidden and rl.hidden must match (configs/bc/robomimic_bc.yaml, "
            "configs/rl/ppo.yaml).")
    with torch.no_grad():
        for a, b in zip(src, dst):
            if a.weight.shape != b.weight.shape:
                raise RuntimeError(f"layer shape mismatch {tuple(a.weight.shape)} vs "
                                   f"{tuple(b.weight.shape)} -- see bc.hidden / rl.hidden")
            b.weight.copy_(a.weight)
            b.bias.copy_(a.bias)
    return len(src)


def verify_bc_transfer(model, ckpt_path: str, device, samples: int = 512) -> float:
    """max |PPO_actor(obs) - BC(obs)| over random observations.

    The transfer is the premise of this phase; measuring it turns "we
    initialised from BC" into a number that lands in the MLflow run.
    """
    bc = _bc_actor(ckpt_path, device)
    x = torch.randn(samples, OBS_DIM, device=device)
    with torch.no_grad():
        want = bc({"flat": x})
        got = _actor_module(model)(model.policy.features_extractor(x))
    return float((want - got).abs().max())


def make_ppo(venv, cfg, device, tb_dir: str) -> StabilisedPPO:
    r = cfg.rl
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[str(r.activation)]
    n_steps = int(r.rollout_steps)
    batch = max(1, venv.num_envs * n_steps // int(r.minibatches))
    model = StabilisedPPO(
        "MlpPolicy", venv, device=device, verbose=0, seed=int(cfg.seed),
        learning_rate=float(r.lr), n_steps=n_steps, batch_size=batch,
        n_epochs=int(r.epochs), gamma=float(r.gamma), gae_lambda=float(r.lam),
        clip_range=float(r.clip), ent_coef=float(r.entropy_coef),
        vf_coef=float(r.value_coef), max_grad_norm=float(r.max_grad_norm),
        target_kl=float(r.target_kl), tensorboard_log=tb_dir,
        policy_kwargs=dict(net_arch=dict(pi=list(r.hidden), vf=list(r.hidden)),
                           activation_fn=act, features_extractor_class=FlattenExtractor,
                           log_std_init=float(np.log(float(r.init_noise_std)))))
    if str(r.output_activation) == "tanh":
        # robomimic's BC head squashes with tanh. Giving the PPO actor the same
        # head is what makes the transferred weights reproduce the BC policy
        # EXACTLY -- without it the initialisation is merely "similar", which is
        # not something verify_bc_transfer could check.
        model.policy.action_net = nn.Sequential(model.policy.action_net, nn.Tanh()).to(device)
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(), lr=float(r.lr), **model.policy.optimizer_kwargs)
    return model


# ---------------------------------------------------------------------------
def train_ppo(cfg, args_cli, make_env, evaluate_policy) -> None:
    import mlflow

    set_seed(int(cfg.seed))
    wandb_env(cfg)
    r = cfg.rl
    runs_dir = os.path.join(REPO_ROOT, "artifacts", "runs", "arm_bcrl006_rl")
    tb_dir = os.path.join(str(cfg.mlops.tensorboard.dir), "ppo")
    deploy_dir = os.path.join(REPO_ROOT, "artifacts", "deploy")
    eval_dir = os.path.join(REPO_ROOT, "artifacts", "eval")
    for d in (runs_dir, tb_dir, deploy_dir, eval_dir):
        os.makedirs(d, exist_ok=True)

    env = make_env(int(r.num_envs), False)
    venv = MjVecEnv(env)
    if bool(r.reward_normalization):
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                            clip_reward=float(r.reward_norm_clip), gamma=float(r.gamma))
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    model = make_ppo(venv, cfg, device, tb_dir)

    bc_ckpt, anchor = None, None
    if bool(r.bc_init):
        bc_ckpt = find_bc_checkpoint()
        n = init_actor_from_bc(model, bc_ckpt, device)
        err = verify_bc_transfer(model, bc_ckpt, device)
        print(f"[rl] initialised actor from BC ({n} layers): {bc_ckpt}")
        print(f"[rl] BC transfer check: max |PPO(obs) - BC(obs)| = {err:.3e}")
        if err > 1e-3:
            raise RuntimeError(
                f"the PPO actor does not reproduce the BC policy (max diff {err:.3e}). "
                "The initialisation is the premise of this phase -- fix the layer "
                "shapes / activations before spending the compute.")
        # the reference must be the actor AS TRANSFERRED: snapshotting it after
        # training starts anchors to an already-degraded policy
        reference = copy.deepcopy(_actor_module(model)).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        anchor = BCAnchor(_actor_module(model), reference, lr=float(r.bc_anchor_lr),
                          coef=float(r.bc_anchor_coef),
                          decay_iters=int(r.bc_anchor_decay_iters),
                          minibatch=int(r.bc_anchor_minibatch),
                          max_grad_norm=float(r.max_grad_norm),
                          start_iter=int(r.critic_warmup_iters),
                          min_coef=float(r.bc_anchor_min_coef))
        print(f"[rl] BC anchor: coef {float(r.bc_anchor_coef)} -> "
              f"{float(r.bc_anchor_min_coef)} over {int(r.bc_anchor_decay_iters)} iters, "
              f"after the {int(r.critic_warmup_iters)}-iteration critic warm-up")
    else:
        print("[rl] rl.bc_init=false -- PPO from scratch (ablation)")
    model.setup_stabilisers(int(r.critic_warmup_iters), anchor)

    steps_per_iter = env.num_envs * int(r.rollout_steps)
    total_steps = steps_per_iter * int(r.total_iterations)
    callbacks = [PolicyCheckpoint(runs_dir, int(r.save_interval), int(r.rollout_steps),
                                  env.num_envs)]

    with mlflow_run(cfg, run_name=f"{cfg.experiment_name}_rl"), \
            wandb_run(cfg, run_name=f"{cfg.experiment_name}_rl") as wb:
        mlflow.log_param("rl_total_env_steps", total_steps)
        if wb is not None:
            callbacks.append(WandbScalars(wb))
        print(f"[rl] PPO: {env.num_envs} envs x {int(r.rollout_steps)} steps x "
              f"{int(r.total_iterations)} iters = {total_steps} env-steps "
              f"(tensorboard: {tb_dir})", flush=True)
        model.learn(total_timesteps=total_steps, callback=callbacks, tb_log_name="ppo")
        final = os.path.join(runs_dir, "model_final.pt")
        torch.save(model.policy.state_dict(), final)

        # RL does not improve monotonically. Sweep the saved checkpoints in
        # CLOSED LOOP and ship the best -- ARM_BCRL005 learned this the
        # expensive way, by shipping a final iteration worse than the BC policy
        # it started from.
        ckpt_path = final
        if bool(r.select_best_checkpoint):
            ckpt_path = _sweep_checkpoints(model, env, runs_dir, cfg, eval_dir,
                                           evaluate_policy)
        model.policy.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True))
        metrics = _evaluate_and_publish(model, env, cfg, ckpt_path, bc_ckpt,
                                        deploy_dir, eval_dir, evaluate_policy, wb)

    with open(os.path.join(REPO_ROOT, "artifacts", "result.json"), "w", encoding="utf-8") as f:
        json.dump({"success_rate": metrics["success_rate"],
                   "mean_max_lift": metrics["mean_max_lift"]}, f, indent=2)


def _policy_act_fn(model):
    def act_fn(obs):
        a, _ = model.predict(obs.numpy(), deterministic=True)
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return act_fn


def _sweep_checkpoints(model, env, runs_dir: str, cfg, eval_dir: str, evaluate_policy) -> str:
    import mlflow
    paths = sorted(glob.glob(os.path.join(runs_dir, "model_*.pt")))
    if not paths:
        return os.path.join(runs_dir, "model_final.pt")
    device = model.device
    rows = []
    print(f"[rl] checkpoint sweep: {len(paths)} checkpoints x "
          f"{int(cfg.eval.episodes) * env.num_envs} episodes", flush=True)
    for p in paths:
        model.policy.load_state_dict(
            torch.load(p, map_location=device, weights_only=True))
        # every checkpoint sees the SAME scenes (paired comparison)
        mt = evaluate_policy(env, _policy_act_fn(model), int(cfg.eval.episodes),
                             int(cfg.eval.max_steps), seed=7000 + int(cfg.seed))
        rows.append({"checkpoint": p, **mt})
        print(f"[rl]   {os.path.basename(p):20s} success {mt['success_rate']:.3f} "
              f"place_err {mt['mean_place_err']:.4f}", flush=True)
    best = max(rows, key=lambda x: (x["success_rate"], -x["mean_place_err"]))
    out = os.path.join(eval_dir, "ckpt_sweep.json")
    print("[rl] NOTE: the sweep's own number is the one this checkpoint was "
          "SELECTED by, so it is optimistic (winner's curse over "
          f"{len(rows)} candidates). The reported result below is an "
          "independent re-measurement on a different episode draw.")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "selected": best["checkpoint"]}, f, indent=2)
    mlflow.log_artifact(out)
    mlflow.set_tag("selected_checkpoint", os.path.basename(best["checkpoint"]))
    print(f"[rl] selected {os.path.basename(best['checkpoint'])} "
          f"(success {best['success_rate']:.3f})")
    return best["checkpoint"]


def _evaluate_and_publish(model, env, cfg, ckpt_path, bc_ckpt, deploy_dir, eval_dir,
                          evaluate_policy, wb) -> dict:
    """Evaluate, export ONNX, register -- and REFUSE to overwrite the deployed
    teacher with a policy that measures worse than BC.

    The guard runs BEFORE the ONNX is written. ARM_BCRL005 had it after, so a
    regressed policy still overwrote ``*_rl_policy.onnx`` -- and phase 7 reads
    that FILE, not the registry alias, so the guard reported success while the
    teacher was already poisoned.
    """
    import mlflow

    metrics = evaluate_policy(env, _policy_act_fn(model), int(cfg.eval.episodes),
                              int(cfg.eval.max_steps), seed=9000 + int(cfg.seed))
    mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()
                        if isinstance(v, (int, float)) and v == v})
    mlflow.set_tag("checkpoint_path", ckpt_path)
    mlflow.set_tag("bc_init_checkpoint", str(bc_ckpt))
    if wb is not None:
        wb.log({f"rl_final/{k}": v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v})
    with open(os.path.join(eval_dir, "eval_report_rl_finetune.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    bc_report = os.path.join(eval_dir, "eval_report_bc.json")
    bc_success = None
    if os.path.exists(bc_report):
        with open(bc_report, encoding="utf-8") as f:
            bc_success = json.load(f).get("success_rate")
    tol = float(cfg.rl.get("promote_bc_parity_tol", 0.0))
    regressed = (bool(cfg.rl.get("promote_requires_bc_parity", True))
                 and bc_success is not None
                 and metrics["success_rate"] < bc_success - tol)

    actor = OnnxActor(model.policy).eval().cpu()
    deployed = os.path.join(deploy_dir, "arm_bcrl006_rl_policy.onnx")
    onnx_path = os.path.join(deploy_dir, "arm_bcrl006_rl_policy_REJECTED.onnx") \
        if regressed else deployed
    onnx_export.export_onnx(actor, OBS_DIM, onnx_path)
    val = onnx_export.validate_onnx(actor, OBS_DIM, onnx_path)
    lat = onnx_export.benchmark_latency(actor, OBS_DIM, onnx_path)
    with open(os.path.join(deploy_dir, "deploy_report_rl.json"), "w", encoding="utf-8") as f:
        json.dump({"onnx_path": onnx_path, "validation": val, "latency": lat,
                   "checkpoint": ckpt_path, "bc_init_checkpoint": bc_ckpt,
                   "eval_metrics": metrics}, f, indent=2, ensure_ascii=False)

    version = registry.register_onnx_model(
        onnx_path, "arm_bcrl006_rl_policy",
        {"success_rate": metrics["success_rate"], "mean_max_lift": metrics["mean_max_lift"],
         "onnx_max_abs_diff": val.get("max_abs_diff", -1.0),
         "onnx_latency_ms": lat.get("onnx_ms", -1.0)})
    if regressed:
        kept = deployed if os.path.exists(deployed) else "the BC policy (fallback)"
        print(f"[rl] REGRESSION GUARD: success {metrics['success_rate']:.3f} < BC "
              f"{bc_success:.3f} (tol {tol:g}) -- NOT promoted, and the deployed "
              f"teacher was NOT overwritten. Phase 7 will use {kept}.")
        mlflow.set_tag("rl_regression_vs_bc",
                       f"{metrics['success_rate']:.4f} < {bc_success:.4f}")
    else:
        champ = registry.promote_if_best("arm_bcrl006_rl_policy", version.version,
                                         "success_rate", metrics)
        print(f"[rl] registered arm_bcrl006_rl_policy v{version.version}; champion={champ}")
    print("[rl] final eval metrics:", json.dumps(metrics, indent=2))
    if bc_success is not None:
        print(f"[rl] BC {bc_success:.3f} -> RL {metrics['success_rate']:.3f} "
              f"({metrics['success_rate'] - bc_success:+.3f})")
    return metrics
