# -*- coding: utf-8 -*-
"""モデルレジストリは本物の MLflow Model Registry（ローカルのファイル保存）。

RL002 は ``ModelRegistry`` クラスを自作して（``registry.json`` と
チェックポイントの複製）、「MLflow Model Registry を模した」と言っていた。
RL003 は実物のAPIを呼ぶ。``mlflow.onnx.log_model(...,
registered_model_name=...)`` が自動で新しい版を作り、``MlflowClient``
の別名（``champion``）が常に最良の版を指す。既に MLflow を回している客に
対してモデルの昇格を配線するなら、普通はこうする、という形に
そろえてある。
"""

from __future__ import annotations
import onnx
import mlflow
import mlflow.onnx
from mlflow import MlflowClient


def register_onnx_model(onnx_path: str, name: str, metrics: dict) -> "mlflow.entities.model_registry.ModelVersion":
    """ONNX ファイルを現在の MLflow の run に記録し、Model Registry の
    ``name`` の新しい版として登録する。"""
    model = onnx.load(onnx_path)
    mlflow.log_metrics(metrics)
    info = mlflow.onnx.log_model(model, name="policy", registered_model_name=name)
    client = MlflowClient()
    return client.get_model_version(name, info.registered_model_version)


def promote_if_best(name: str, version: str, metric_key: str, metrics: dict, mode: str = "max",
                    alias: str = "champion") -> bool:
    """``metrics[metric_key]`` を、いま ``alias`` が指しているものと
    ``version`` で比べ、今回の方が良ければ別名を動かす。
    この版が優勝になった（またはそのままだった）なら True を返す。
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
