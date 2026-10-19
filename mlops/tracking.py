# -*- coding: utf-8 -*-
"""実験管理。MLflow + Weights & Biases + TensorBoard。全部ローカル。

3つ使うのは、3つが別の問いに答えるから。3つ入れておいて、どれがどの
問いに答えるのか言えないなら、それは何も入れていないのと同じ、という
考えで分けている:

  MLflow      実行の台帳。パラメータ、指標、成果物、そしてどの ONNX が
              「優勝」かを決めるモデルレジストリ。これが記録の正本で、
              src/infer.py は何を出荷するかをレジストリから読む。
              つまり MLflow は報告用ではなく、荷重を受けている。
  W&B         生の眺め。走っている最中に見る曲線（PPO はロールアウト
              ごと、ACT はエポックごと）に加えて、推論時に書く閉ループ
              評価の表と棒グラフ。1つのプロジェクトのページで
              「本当に良くなったのか」に答えられる。
  TensorBoard デバッグ用。stable-baselines3 が自前で書く反復ごとの PPO
              スカラー（価値損失、エントロピー、説明分散、近似KL）と、
              robomimic 自身の BC 曲線。RL が崩れたときに理由を言える
              解像度を持つのは、3つの中でこれだけ。
              使い分けはここで固定してある。

既定では全部ローカル。MLflow は SQLite ファイル、W&B はオフライン、
TensorBoard はイベントファイル。アカウントもAPIキーもネットワークも
要らない。それが、データを機外に出せない案件でこの構成が使える理由。
クラウドへ流したいなら ``wandb login`` と ``mlops.wandb.mode=online``、
変更はそれだけ。

RL002 は ``ExperimentTracker`` を自作して（manifest.json と metrics.csv）
「MLflow 風」と呼んでいた。RL003 は実物のライブラリを使う:

  * MLflow   ローカルの ``file:`` の tracking URI に対して
                ``mlflow.start_run`` / ``log_params`` / ``log_metrics`` /
                ``log_artifacts``。各 run で解決済みの Hydra 設定を自動保存する。
  * W&B      ``WANDB_MODE=offline`` の環境変数をここで立てるので、自前の
                ``mlflow_run`` の文脈でも、RSL-RL 内蔵の
                ``WandbSummaryWriter``（runner 設定の ``logger: wandb``）でも、
                アカウントもAPIキーも無しで同じローカルの run に書く。
"""

from __future__ import annotations
import os
import platform
import socket
import subprocess
from contextlib import contextmanager

import mlflow
from omegaconf import DictConfig, OmegaConf


def _patch_wandb_working_set() -> None:
    """ARM_BCRL005 から残したもの。綺麗なローカル環境では何もしない。

    Isaac Sim の Python 環境で実際に出るバグの回避。
    ``wandb.util.working_set()``（インストール済みパッケージの出どころを
    記録するために、RSL-RL 内蔵の ``WandbSummaryWriter`` を含む全ての
    ``wandb.init()`` が呼ぶ）は、パッケージごとに
    ``KeyError``/``UnicodeDecodeError``/``TypeError`` しか抑えない。
    Isaac Sim は METADATA を持たない ``pip_prebundle`` の dist-info を
    いくつか同梱していて（例:
    ``isaacsim.robot_motion.pink/pip_prebundle/eigenpy-3.13.0.dist-info``）、
    そこで ``importlib_metadata.MetadataNotFound``（wandb の抑制一覧に
    無い ``FileNotFoundError`` の子）が飛び、最初の反復より前に run が
    落ちる。Isaac Sim の同梱パッケージに手を入れる代わりに、ここで
    （``wandb.init()`` より前に1回だけ）当てる。
    """
    import contextlib
    import wandb.util as _wandb_util

    if getattr(_wandb_util.working_set, "_amr_patched", False):
        return

    def _working_set():
        from importlib.metadata import distributions
        for d in distributions():
            with contextlib.suppress(Exception):
                yield _wandb_util.InstalledDistribution(key=d.metadata["Name"], version=d.version)

    _working_set._amr_patched = True
    _wandb_util.working_set = _working_set


_WANDB_SETTINGS_PATCHED = False


def _patch_wandb_settings() -> None:
    """RSL-RL 内蔵の ``WandbSummaryWriter`` と、この Isaac Sim の Python に
    入っている wandb との、2つ目の非互換の回避。writer は
    ``wandb.Settings(start_method="thread")`` を呼ぶが、wandb は
    ``Settings`` を ``extra="forbid"`` の pydantic モデルに作り替え、
    ``start_method``（指していたサービス起動の切り替え自体が無くなった）を
    落とした。だから呼び出しは
    ``ValidationError: start_method Extra inputs are not permitted`` を
    ``runner.learn()`` から出す。つまり環境、ランナー、BC の重み移植が
    全部立ち上がった後で、PPO のフェーズを殺す。

    回避はここで（``wandb.init()`` より前に1回だけ）、``wandb.Settings``
    のモジュール属性を差し替える。RSL-RL のような外部の呼び出し側が使うのは
    そちらで、wandb の内部はクラスを直接 import している。差し替え先は、
    入っている版が知らないキーワード引数を落とす子クラス。Isaac Sim の
    site-packages には何も持ち込まないし、``start_method`` を今も受け取る
    wandb では何もしない。
    """
    global _WANDB_SETTINGS_PATCHED
    if _WANDB_SETTINGS_PATCHED:
        return

    import wandb

    base = wandb.Settings
    fields = getattr(base, "model_fields", None)
    if not fields:
        # pydantic より前の wandb。未知の設定は wandb 自身が許す。
        _WANDB_SETTINGS_PATCHED = True
        return

    known = set(fields)

    def _drop_unknown(kwargs: dict) -> dict:
        dropped = sorted(k for k in kwargs if k not in known)
        if dropped:
            kwargs = {k: v for k, v in kwargs.items() if k not in dropped}
            print(f"[wandb] ignoring setting(s) removed in wandb "
                  f"{getattr(wandb, '__version__', '?')}: {', '.join(dropped)}")
        return kwargs

    try:
        # 子クラスにしておけば、渡されたものの型を確認する人のために
        # ``isinstance(x, wandb.Settings)`` が動き続ける
        class _CompatSettings(base):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **_drop_unknown(kwargs))

        wandb.Settings = _CompatSettings
    except Exception:
        # pydantic が子クラスを拒んだ場合。素の工場関数に落ちる。それでも
        # 気にすべき唯一の呼び出し側は直る（RSL-RL は wandb.Settings を
        # 構築するだけで、isinstance で見ることはない）
        def _compat_settings(*args, **kwargs):
            return base(*args, **_drop_unknown(kwargs))

        wandb.Settings = _compat_settings
    _WANDB_SETTINGS_PATCHED = True


def wandb_summary_writer_supported() -> bool:
    """RSL-RL の ``WandbSummaryWriter`` が、入っている wandb に対して
    （上の回避を当てたうえで）実際に構築できるなら True。

    RSL-RL のランナーの ``logger:`` を選ぶのに使う。まだ見たことのない
    理由で W&B が使えないとき、PPO の実行はパイプラインの数時間後に死ぬ
    のではなく TensorBoard に落ちる。どちらにしても MLflow が
    パラメータと指標を記録する。
    """
    try:
        _patch_wandb_working_set()
        _patch_wandb_settings()
        import wandb
        wandb.Settings(start_method="thread")
        return True
    except Exception as exc:  # pragma: no cover -- 環境に依存する
        print(f"[wandb] RSL-RL's WandbSummaryWriter is unusable here ({exc!r}); "
              f"falling back to the TensorBoard logger for this run")
        return False


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10)
        return out.decode().strip().splitlines()[0]
    except Exception:
        return "unknown"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def wandb_env(cfg: DictConfig) -> None:
    """``wandb.init()``（自前も RSL-RL のものも）より前に W&B の環境変数を立てる。

    既定はオフライン。本物の W&B の run を ``mlops.wandb.dir`` に書き、
    外部への通信もAPIキーも無し。``wandb login`` を1回して
    ``mlops.wandb.mode=online`` にすれば、後からクラウドへ同期できる。
    コードの変更は要らない。
    """
    _patch_wandb_working_set()
    _patch_wandb_settings()
    if not cfg.mlops.wandb.enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return
    os.makedirs(cfg.mlops.wandb.dir, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", cfg.mlops.wandb.mode)
    os.environ.setdefault("WANDB_DIR", cfg.mlops.wandb.dir)
    os.environ.setdefault("WANDB_PROJECT", cfg.mlops.wandb.project)


@contextmanager
def mlflow_run(cfg: DictConfig, run_name: str):
    """ローカルのファイル保存の tracking URI に対して MLflow の run を開き、
    解決済みの Hydra 設定と環境の出どころを、パラメータとタグとして記録する。
    """
    mlflow.set_tracking_uri(cfg.mlops.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlops.mlflow.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        params = _flatten(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.set_tags({
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "gpu": _gpu_name(),
        })
        yield run


@contextmanager
def wandb_run(cfg: DictConfig, run_name: str):
    """自前の W&B の run を開く。W&B の書き出しを内蔵していない学習ループ、
    つまり ``mlops.wandb`` の設定を渡す先が無いもののため
    （例: robomimic の BC 学習器。``scripts/train_bc_robomimic.py`` 参照）。

    robomimic にも W&B のロガーはある
    （``config.experiment.logging.log_wandb``）が、
    ``robomimic.macros.WANDB_ENTITY`` が ``robomimic/scripts/setup_macros.py``
    で事前設定されていることを強く要求し、init の失敗を30秒おきに10回
    再試行する。このプロジェクトが依存したくない脆い仕掛け。だからその
    経路は切り、同じ情報（パラメータと最終指標）を素の
    ``wandb.init()``/``wandb.log()``/``wandb.finish()`` でここから記録する。
    上の ``mlflow_run`` と同じやり方。
    """
    _patch_wandb_working_set()
    _patch_wandb_settings()
    if not cfg.mlops.wandb.enabled or cfg.mlops.wandb.mode == "disabled":
        yield None
        return
    import wandb
    os.makedirs(cfg.mlops.wandb.dir, exist_ok=True)
    run = wandb.init(project=cfg.mlops.wandb.project, name=run_name, dir=cfg.mlops.wandb.dir,
                     mode=cfg.mlops.wandb.mode, config=_flatten(OmegaConf.to_container(cfg, resolve=True)),
                     reinit="finish_previous")
    try:
        yield run
    finally:
        wandb.finish()


# ---------------------------------------------------------------------------
# ARM_BCRL006 での追加。TensorBoard と、評価／推論時の W&B。
# ---------------------------------------------------------------------------
@contextmanager
def tensorboard_writer(cfg: DictConfig, run_name: str):
    """``mlops.tensorboard.dir`` の下の SummaryWriter。無効なら None。

    stable-baselines3 は自分で TensorBoard のスカラーを書くので、これは
    書かないフェーズ（BC と視覚運動方策の学習器）のため。``tensorboard
    --logdir artifacts`` で、パイプラインの全フェーズが1ページに出る。
    """
    tb = cfg.mlops.get("tensorboard", None)
    if tb is None or not bool(tb.enabled):
        yield None
        return
    from torch.utils.tensorboard import SummaryWriter
    path = os.path.join(str(tb.dir), run_name)
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path)
    try:
        yield writer
    finally:
        writer.flush()
        writer.close()


def wandb_log_eval(cfg: DictConfig, run_name: str, results: dict,
                   metrics: tuple = ("success_rate", "mean_max_lift",
                                     "mean_place_err", "mean_reward")) -> None:
    """閉ループの評価を、表と棒グラフ1枚として W&B に記録する。

    ``results``: {手法名: {指標: 値}}。

    学習曲線は損失が下がったことを教えてくれるが、ロボットが仕事をするか
    は言えない。それには閉ループの数値が必要で、その数値は全ての学習器が
    終わった後、推論時に出る。曲線と同じ W&B のプロジェクトへ送ることが、
    関係者が聞く唯一の問いに、プロジェクトのページで答えられる条件。
    だから評価の数値もここへ送る。
    """
    if not cfg.mlops.wandb.enabled or cfg.mlops.wandb.mode == "disabled":
        return
    _patch_wandb_working_set()
    _patch_wandb_settings()
    import wandb
    os.makedirs(cfg.mlops.wandb.dir, exist_ok=True)
    run = wandb.init(project=cfg.mlops.wandb.project, name=run_name,
                     dir=cfg.mlops.wandb.dir, mode=cfg.mlops.wandb.mode,
                     job_type="evaluation", reinit="finish_previous")
    try:
        cols = ["method"] + list(metrics)
        table = wandb.Table(columns=cols)
        for name, m in results.items():
            table.add_data(name, *[float(m.get(k, float("nan"))) for k in metrics])
        run.log({"closed_loop/table": table})
        for k in metrics:
            run.log({f"closed_loop/{k}": wandb.plot.bar(
                wandb.Table(data=[[n, float(m.get(k, 0.0))] for n, m in results.items()],
                            columns=["method", k]),
                "method", k, title=k)})
        for name, m in results.items():
            run.summary.update({f"{name}/{k}": v for k, v in m.items()
                                if isinstance(v, (int, float)) and v == v})
    finally:
        wandb.finish()


def import_tensorboard_curves_to_wandb(cfg: DictConfig, event_dir: str,
                                       run_name: str, prefix: str = "") -> int:
    """TensorBoard のイベントのディレクトリを、W&B の run へ流し込む。

    robomimic は自分の BC の学習ループを持っていて、エポックごとのフックを
    出さない。ただし TensorBoard のイベントは書く。学習器が終わった後で
    それを読み直せば、BC の曲線が PPO と ACT の曲線と同じ W&B のページに
    乗る。「見る場所は1つ」を実際に成り立たせるのはそれ。最善努力で、
    イベントファイルが無い・読めないことが学習を失敗させてはならない。
    """
    if not cfg.mlops.wandb.enabled or cfg.mlops.wandb.mode == "disabled":
        return 0
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import wandb
        acc = EventAccumulator(event_dir, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if not tags:
            return 0
        series = {t: acc.Scalars(t) for t in tags}
        steps = sorted({e.step for t in tags for e in series[t]})
        os.makedirs(cfg.mlops.wandb.dir, exist_ok=True)
        run = wandb.init(project=cfg.mlops.wandb.project, name=run_name,
                         dir=cfg.mlops.wandb.dir, mode=cfg.mlops.wandb.mode,
                         job_type="training", reinit="finish_previous")
        by_step = {s: {} for s in steps}
        for t in tags:
            for e in series[t]:
                by_step[e.step][f"{prefix}{t}"] = float(e.value)
        for s in steps:
            run.log(by_step[s], step=int(s))
        wandb.finish()
        return len(steps)
    except Exception as exc:                        # pragma: no cover
        print(f"[wandb] could not import TensorBoard curves from {event_dir}: {exc!r}")
        return 0
