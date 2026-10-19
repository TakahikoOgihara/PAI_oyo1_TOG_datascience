# -*- coding: utf-8 -*-
"""フェーズ6。BC 方策から始める PPO 微調整（stable-baselines3）。

このフェーズが要る理由。BC は教師役で上限が決まる。スクリプト
エキスパートが天井で、複製した方策は複製元より必ず悪い。一方で PPO を
ゼロから回すと、デモが既に知っていたことを再発見するのに何時間も
燃やす。BC の重みから初期化すれば両方取れる。そして差は実測する
（src/rl_finetune.py が2つの評価結果を比べる）。主張では済ませない。
``rl.bc_init=false`` にすれば、初期化が何を買っているかを正確に
数値化できる。

4つの安定化策の理由。ARM_BCRL005 の最初の本番は、成功率0.606の BC
方策が600反復で 0.000 まで崩れるのを見ることになった。報酬ハックでも
エントロピー崩壊でもなく、クリティックが発散した。ランダム初期化の
価値ヘッドが BC 方策の 800〜1900 のリターンに当たって価値損失 4e4 を
出し、``advantage = returns - values`` はノイズになり、PPO は忠実に
それを追った。方策が成功しなくなると疎な成功勾配は消え、二度と
戻らない。効いた順に対処を挙げる:

  1. リターン正規化。クリティックの回帰目標を O(1) にする。
     ``VecNormalize(norm_reward=True)`` は、ARM_BCRL005 で手書きしたものと
     同じことをするライブラリ自身の実装。
  2. クリティックのウォームアップ。最初のN反復はアクターを凍結し、
     方策を動かす前に価値関数を追いつかせる。
  3. 学習率を小さくする。3e-4 はゼロから回すときの値。
  4. BC アンカー。ロールアウトが実際に通った状態の上で、アクターを BC の
     スナップショットへ引き戻す補助的な勾配ステップ。0へ減衰させるので、
     RL は最終的に BC を超えられる。RLHF の KL-to-SFT 罰則、DAPG の BC
     勾配項（Rajeswaran et al. 2018）と同じ考え方。

減衰させるのはアンカーの最適化器の学習率で、損失の係数ではない。
この区別は ARM_BCRL005 で1本の実行を潰した。Adam の更新
``lr * m / (sqrt(v) + eps)`` は損失の定数倍に対して不変なので、係数を
1.0 -> 0.0 と落としても、正確に0になる瞬間まで何も起きず、そこで
制約が崖のように消える。減衰させたいのは歩幅の方。

RSL-RL（ARM_BCRL005 の選択）ではなく stable-baselines3 を使う理由。
RSL-RL のランナーは Isaac Lab のベクトル化環境ラッパー前提で書かれて
いて、その外では使えない。SB3 は汎用RLライブラリの主流で、``VecEnv``
の契約は安定して文書化されており、TensorBoard のスカラーも自分で書く。
アルゴリズムは同じクリップ目的の PPO。
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
    """``MuJoCoPickPlaceEnv`` に対する SB3 の ``VecEnv``。

    環境はすでにベクトル化されていて自動リセットもするので、これは
    ラッパーの積み重ねではなく薄い変換。SubprocVecEnv も、環境ごとの
    Python オブジェクトも無く、バッチ全体で ``mj_step`` が1回。
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
            # SB3 は打ち切り（truncated）エピソードの価値をブートストラップし、
            # 終了（terminated）ではしない。ここでの成功は真の終端状態で、
            # タイムアウトはそうではない。2つを取り違えると、ホライズン付近の
            # リターン推定が全部偏る。
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
    """BC のスナップショットへ引き戻す補助的な勾配ステップ（詳細は冒頭）。"""

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
        # 専用の最適化器を持たせる。アンカーが PPO のスケジュールを乱して
        # はならず、PPO の適応的な学習率がアンカーの減衰を乱してもならない
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
        for g in self.optimizer.param_groups:      # 減衰させるのは歩幅
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
    """PPO + クリティックのウォームアップ + BC アンカー。他は素の SB3。"""

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
            # ロールアウトが実際に通った状態でアンカーする。ランダムな観測で
            # 方策を縛るのは、絶対に行かない場所で縛ること
            obs = torch.as_tensor(
                self.rollout_buffer.observations.reshape(-1, OBS_DIM),
                dtype=torch.float32, device=self.device)
            self.logger.record("train/bc_anchor_mse", self.anchor.step(obs, self.iteration))
            self.logger.record("train/bc_anchor_coef", self.anchor.coef_at(self.iteration))
        self.iteration += 1


class PolicyCheckpoint(BaseCallback):
    """SB3 の ``.zip`` ではなく、方策の state dict を保存する。

    アクターのヘッドは構築後に書き換えている（robomimic の押し潰した
    出力に合わせて tanh を足す）ので、``PPO.load`` はパラメータ名の合わない
    方策を組み立ててしまう。すでに書き換え済みの同じ方策オブジェクトへ
    state dict を読み込むならその問題は起きないし、チェックポイントの
    総当たり評価に必要なのもそちら。
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
    """SB3 のロガーを W&B に写して、学習を後からではなく進行中に見られる
    ようにする。TensorBoard は同じスカラーを自前で受け（SB3 が自分で
    書く）、MLflow は最後に要約を受ける。3つの道具に3つの仕事
    （configs/mlops/default.yaml）。"""

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
    """決定的な方策グラフ。obs -> action。実際に出荷するもの。"""

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
    """robomimic の BC の MLP を、層ごとに PPO のアクターへ写す。

    形が同じであることが条件で、``bc.hidden == rl.hidden`` と
    ``rl.activation=relu`` がそれを保証する。違っていれば設定の間違いなので、
    黙って半分だけ写すのではなく例外にする。
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
    """ランダムな観測に対する max |PPO_actor(obs) - BC(obs)|。

    重みの移植がこのフェーズの前提なので、それを測れば「BC から
    初期化した」が MLflow の run に載る数値になる。
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
        # robomimic の BC ヘッドは tanh で押し潰す。PPO のアクターに同じ
        # ヘッドを付けることが、移植した重みが BC 方策を正確に再現する
        # 条件。付けないと初期化は「似ている」だけになり、
        # verify_bc_transfer が確認できるものではなくなる。
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
        # 基準にするのは「移植した直後のアクター」。学習が始まった後で
        # スナップショットを取ると、すでに劣化した方策に縛ることになる
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

        # RL は単調に良くならない。保存したチェックポイントを閉ループで
        # 総当たりし、最良を出荷する。ARM_BCRL005 は、出発点の BC 方策より
        # 悪い最終反復を出荷して、これを高い授業料で学んだ。
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
        # どのチェックポイントも同じシーンを見る（対応のある比較）
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
    """評価して ONNX を書き出して登録する。そして BC より悪い方策で
    配備済みの教師を上書きすることを拒否する。

    ガードは ONNX を書く前に走る。ARM_BCRL005 では後ろにあったので、
    劣化した方策でも ``*_rl_policy.onnx`` を上書きしてしまっていた。
    フェーズ7が読むのはレジストリの別名ではなくそのファイルなので、
    ガードは成功を報告しつつ、教師はすでに汚染されていた。
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
