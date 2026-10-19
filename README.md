# myCobot 280 Pi の pick & place を MuJoCo で学習させる

6軸アーム myCobot 280 Pi（＋平行2指グリッパ）に、机の上の赤い円柱ワーク（直径30mm）を
掴んで5cm横へ運ぶ動作を学習させる。シミュレータは MuJoCo、学習はノートPC 1台のCPUで回る。
最後に Jetson Orin Nano + Raspberry Pi 向けのエッジ配布物まで作る。

学習は3段構え。スクリプトIKのエキスパートでデモを集め、模倣学習（BC）で土台を作り、
PPOで微調整し、蒸留して手首カメラ画像だけで動く方策（ACT）にする。
あわせて「デモを生成で増やす（MimicGen）」「物理をばらす（ドメインランダム化）」が
本当に効くのかを 2×2・各3シードで実測する。

## 動かし方

```bash
# 1. 環境構築（Python 3.10 の venv を作って依存を入れる。10分ほど）
bash setup_env.sh
source ~/.venvs/arm_bcrl006/bin/activate

# 2. 全11フェーズを通す（約1時間、CPUのみでも完走する）
bash run_all.sh

# 3. 2x2 の比較実験（約80分）
bash run_ablation.sh

# 4. 動作を見る
python mujoco_sim/train_arm_mujoco.py --mode play   # ビューアで再生
python src/record_video.py                          # 動画に書き出す

# 5. テスト（全てCPU、数秒）
pytest -q tests
```

GPU は不要。`device=cpu` のまま全フェーズ動く（ACT の学習だけ遅くなる）。
設定は Hydra なので、コマンドラインから上書きできる。

```bash
python src/collect_data.py env.num_envs=8          # 1項目だけ変える
bash run_all.sh mimicgen.enable=true               # フェーズ1.5を有効化
```

## パイプライン（run_all.sh が順に実行する）

| # | フェーズ | 中身 |
|---|---|---|
| 0 | assets | FK定数から myCobot の MJCF を生成（メッシュのダウンロード不要） |
| 1 | collect | スクリプトIKエキスパートがデモを収集、手首カメラも記録 |
| 1.5 | mimicgen | 既定ではOFF。ONにするとデモを新しいシーンへ転写して増やす |
| 2 | preprocess | 外れ値除去と robomimic 形式への変換、画質ゲート |
| 3 | BC | robomimic の BC を学習、ONNX と MLflow に登録 |
| 4 | eval | ランダム／エキスパート／BC の閉ループ評価 |
| 5 | LeRobot | データセット書き出し（任意、無ければ自動スキップ） |
| 6 | RL | BCの重みから PPO で微調整 |
| 7 | distill | RLの方策で手首カメラ付きデモを再収集 |
| 8 | visuomotor | ACT（既定）または Diffusion Policy を学習、ONNX化 |
| 9 | eval | 画像方策の閉ループ評価 |
| 10 | deploy | リリースゲートを通ったものだけエッジ配布物を作る |

フェーズ2と10はゲートになっていて、基準を満たさないとそこで止まる。
手首カメラの画像が使い物にならないデータセット、しきい値に届かない方策は先へ進まない。

## ディレクトリ

```
assets/        MJCF生成スクリプトと参考写真
arm_common/    エキスパートの状態機械、FK、ヤコビアン、DLS-IK
mujoco_sim/    環境、ドメインランダム化、PPO、ビューア、エントリポイント
mimicgen/      サブタスク分割とデモ生成
bc_hdf5/       robomimic のデータセット定義
arm_policy/    ACT と Diffusion Policy
scripts/       BC と視覚運動方策の学習スクリプト
src/           各フェーズの実体、比較実験、動画書き出し
deploy/raspi/  実機の安全層とFK（Raspberry Pi 側）
deploy/jetson/ TensorRT ランタイム（Jetson 側）
deploy/operator_pc/  ブラウザ操作コンソール
configs/       Hydra の設定ツリー
tests/         CPUだけで数秒で終わるテスト
```

出力はすべて `artifacts/` に入る。再生成できるのでバージョン管理していない。
`bash run_all.sh` を流せば中身は作り直せる。
