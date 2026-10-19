# -*- coding: utf-8 -*-
"""Model registry = the real MLflow Model Registry (local file store).

RL002 hand-rolled a ``ModelRegistry`` class (``registry.json`` + copied
checkpoint files) and called it "mirrors MLflow Model Registry". RL003 calls
the actual MLflow Model Registry API: ``mlflow.onnx.log_model(...,
registered_model_name=...)`` creates a new version automatically, and a
``MlflowClient`` alias (``champion``) always points at the best one -- this
is exactly how a consultant would wire model promotion for a client that
already runs MLflow.
"""

from __future__ import annotations
import onnx
import mlflow
import mlflow.onnx
from mlflow import MlflowClient


def register_onnx_model(onnx_path: str, name: str, metrics: dict) -> "mlflow.entities.model_registry.ModelVersion":
    """Log the ONNX file to the current MLflow run and register it as a new
    version of ``name`` in the Model Registry."""
    model = onnx.load(onnx_path)
    mlflow.log_metrics(metrics)
    info = mlflow.onnx.log_model(model, name="policy", registered_model_name=name)
    client = MlflowClient()
    return client.get_model_version(name, info.registered_model_version)


def promote_if_best(name: str, version: str, metric_key: str, metrics: dict, mode: str = "max",
                    alias: str = "champion") -> bool:
    """Compare ``metrics[metric_key]`` for ``version`` against whatever the
    current ``alias`` points at, and move the alias if this run is better.
    Returns True if this version became (or stayed) the champion.
    """
    client = MlflowClient()
    candidate = metrics[metric_key]
    try:
        current = client.get_model_version_by_alias(name, alias)
        current_run = client.get_run(current.run_id)
        current_metric = current_run.data.metrics.get(metric_key)
    except Exception:
        current_metric = None

    is_better = (current_metric is None or
                (candidate > current_metric if mode == "max" else candidate < current_metric))
    if is_better:
        client.set_registered_model_alias(name, alias, version)
    return is_better


def champion_uri(name: str, alias: str = "champion") -> str:
    return f"models:/{name}@{alias}"
