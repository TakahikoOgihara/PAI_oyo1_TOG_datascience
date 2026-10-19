# -*- coding: utf-8 -*-
"""Experiment tracking: MLflow + Weights & Biases + TensorBoard, all local.

Three tools because they answer three different questions, and a consultant who
installs all three without being able to say which question each one answers has
installed nothing:

  MLflow      the RUN LEDGER -- params, metrics, artifacts, and the Model
              Registry that decides which ONNX is "champion". This is the
              system of record: src/infer.py reads the registry to pick what
              ships, so MLflow is load-bearing, not reporting.
  W&B         the LIVE VIEW -- curves you watch while a run is in flight
              (PPO per rollout, ACT per epoch), plus a closed-loop evaluation
              table and bar chart written at INFERENCE time, so one project
              page answers "did it actually get better".
  TensorBoard the DEBUG VIEW -- per-iteration PPO scalars (value loss, entropy,
              explained variance, approx KL) written natively by
              stable-baselines3, and robomimic's own BC curves. When RL
              collapses, this is the only one of the three with the resolution
              to say why.

Everything is local by default: MLflow on a SQLite file, W&B in offline mode,
TensorBoard on the event files. No account, no API key, no network -- which is
what makes the stack usable on a client engagement where data cannot leave the
machine. ``wandb login`` + ``mlops.wandb.mode=online`` is the only change needed
to stream to the cloud instead.

RL002 hand-rolled its own ``ExperimentTracker`` (manifest.json + metrics.csv)
and called it "MLflow-style". RL003 uses the actual libraries instead:

  * MLflow   -- ``mlflow.start_run`` / ``log_params`` / ``log_metrics`` /
                ``log_artifacts`` against a local ``file:`` tracking URI.
                Every run auto-snapshots the fully-resolved Hydra config.
  * W&B      -- ``WANDB_MODE=offline`` env vars set here so BOTH our own
                ``mlflow_run`` context AND RSL-RL's built-in
                ``WandbSummaryWriter`` (``logger: wandb`` in the runner cfg)
                write to the same local run without any account/API key.
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
    """Kept from ARM_BCRL005; a no-op on a clean local install.

    Work around a real bug triggered by an Isaac Sim Python environment:
    ``wandb.util.working_set()`` (called by every ``wandb.init()``, including
    RSL-RL's built-in ``WandbSummaryWriter``, to log installed-package
    provenance) only suppresses ``KeyError``/``UnicodeDecodeError``/``TypeError``
    per-package. Isaac Sim ships a few extension-bundled ``pip_prebundle``
    dist-info directories with no METADATA file at all (e.g.
    ``isaacsim.robot_motion.pink/pip_prebundle/eigenpy-3.13.0.dist-info``),
    which raises ``importlib_metadata.MetadataNotFound`` -- a ``FileNotFoundError``
    subclass wandb's suppress list does not cover -- and crashes the run before
    the first training iteration. Patched here (once, before any
    ``wandb.init()``) to also tolerate that, rather than vendoring a fix into
    Isaac Sim's bundled packages.
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
    """Work around a second incompatibility between RSL-RL's built-in
    ``WandbSummaryWriter`` and the wandb shipped in this Isaac Sim Python:
    the writer calls ``wandb.Settings(start_method="thread")``, but wandb
    rebuilt ``Settings`` as a pydantic model with ``extra="forbid"`` and
    dropped ``start_method`` (the service-launch knob it named is gone). The
    call therefore raises
    ``ValidationError: start_method Extra inputs are not permitted`` from
    ``runner.learn()`` -- i.e. AFTER the env, the runner and the BC weight
    transfer are all up -- and kills the PPO phase.

    Patched here (once, before any ``wandb.init()``) by replacing the
    ``wandb.Settings`` module attribute -- the one third-party callers such as
    RSL-RL use; wandb's own internals import the class directly -- with a
    subclass that drops keyword arguments the installed version no longer
    knows about. Nothing is vendored into Isaac Sim's site-packages, and the
    patch is a no-op on a wandb that still accepts ``start_method``.
    """
    global _WANDB_SETTINGS_PATCHED
    if _WANDB_SETTINGS_PATCHED:
        return

    import wandb

    base = wandb.Settings
    fields = getattr(base, "model_fields", None)
    if not fields:
        # pre-pydantic wandb: unknown settings are tolerated by wandb itself.
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
        # a subclass keeps ``isinstance(x, wandb.Settings)`` working for anyone
        # who checks the type of what they were handed
        class _CompatSettings(base):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **_drop_unknown(kwargs))

        wandb.Settings = _CompatSettings
    except Exception:
        # pydantic refused the subclass: fall back to a plain factory, which
        # still fixes the only caller we care about (RSL-RL constructs, never
        # isinstance-checks, wandb.Settings)
        def _compat_settings(*args, **kwargs):
            return base(*args, **_drop_unknown(kwargs))

        wandb.Settings = _compat_settings
    _WANDB_SETTINGS_PATCHED = True


def wandb_summary_writer_supported() -> bool:
    """True when RSL-RL's ``WandbSummaryWriter`` can actually be constructed
    against the installed wandb (after the patches above).

    Used to pick ``logger:`` for the RSL-RL runner: if W&B turns out to be
    unusable for a reason we have not seen yet, the PPO run falls back to
    TensorBoard instead of dying hours into a pipeline -- MLflow records the
    params/metrics either way.
    """
    try:
        _patch_wandb_working_set()
        _patch_wandb_settings()
        import wandb
        wandb.Settings(start_method="thread")
        return True
    except Exception as exc:  # pragma: no cover -- environment-dependent
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
    """Configure W&B env vars before any ``wandb.init()`` (ours or RSL-RL's).

    Offline by default: a real W&B run is written to ``mlops.wandb.dir`` with
    no external network call / API key. Run ``wandb login`` once and set
    ``mlops.wandb.mode=online`` to sync it to the cloud later -- no code
    changes required.
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
    """Open an MLflow run against the local file-store tracking URI and log
    the fully-resolved Hydra config + environment provenance as params/tags.
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
    """Open our own W&B run for training loops that have no built-in W&B
    writer of their own to hand ``mlops.wandb`` config to (e.g. robomimic's
    BC trainer, see ``scripts/train_bc_robomimic.py``).

    robomimic *does* have a built-in W&B logger
    (``config.experiment.logging.log_wandb``), but it hard-asserts on
    ``robomimic.macros.WANDB_ENTITY`` being pre-configured via
    ``robomimic/scripts/setup_macros.py`` and retries failed inits 10x with a
    30s sleep between attempts -- fragile machinery this project does not
    want to depend on. We disable that path and log the same run information
    (params + final metrics) through a plain ``wandb.init()``/``wandb.log()``/
    ``wandb.finish()`` here instead, exactly like ``mlflow_run`` above.
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
# ARM_BCRL006 additions: TensorBoard, and W&B at EVALUATION / INFERENCE time.
# ---------------------------------------------------------------------------
@contextmanager
def tensorboard_writer(cfg: DictConfig, run_name: str):
    """A SummaryWriter under ``mlops.tensorboard.dir``, or None when disabled.

    stable-baselines3 writes its own TensorBoard scalars, so this is for the
    phases that do not: the BC and visuomotor trainers. One ``tensorboard
    --logdir artifacts`` then shows every phase of the pipeline on one page.
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
    """Log a CLOSED-LOOP evaluation to W&B as a table plus one bar chart panel.

    ``results``: {method_name: {metric: value}}.

    Training curves tell you a loss went down. They cannot tell you whether the
    robot does the task -- for that you need the closed-loop numbers, and those
    are produced at inference time, after every trainer has exited. Sending
    them to the same W&B project as the curves is what makes the project page
    answer the only question a stakeholder asks.
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
    """Replay a TensorBoard event directory into a W&B run.

    robomimic owns its BC training loop and offers no per-epoch hook, but it
    does write TensorBoard events. Reading them back after the trainer exits
    puts the BC curve on the same W&B page as the PPO and ACT curves, which is
    what "one place to look" actually requires. Best-effort: a missing or
    unreadable event file must never fail a training run.
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
