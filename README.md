# Reservoir Dynamics

任意の力学系をリザバーとして分析・調整するための研究用ツールである。
自律、入力駆動、閉ループのダイナミクスを区別し、アトラクタ、吸引域、
遷移、条件付き安定性、情報処理容量を同一の証拠規約で扱う。

## 現在の実装範囲

- 吸引確率とWilson信頼区間によるbasin stability推定
- Shannon entropyに基づく実効レパートリー
- 同一入力を受ける複製軌道のpairwise RMS距離
- 有限時間条件付きLyapunov指数
- 共通入力を複数初期状態へ与える離散時間シミュレータ
- スカラーtanhリザバーの収縮上界とground-truth実験
- 局所安定性と大域的replica同期を分離するparameter sweep
- 多次元tanh RNNとJacobian-vector product
- 最大条件付きLyapunov指数と線形遅延記憶曲線
- 安定性、replica同期、記憶容量を同時測定する36条件pilot
- 参照replicaで学習した固定readoutの初期状態間移送容量
- 30 seed bootstrap検証と事前登録判定
- core–reserve block分解とreserve-only不変更新
- reserve feedbackによるcore偏差の決定論的上界
- core保持と新規容量を比較する30 seed plastic-reserve実験
- scalar双安定coreのロバスト正不変区間とsaddle-node forcing margin
- cue形成reserveとcore attractor survivabilityを比較する30 seed実験
- 非対角tanh RNNの符号orthant別robust hyperbox certificate
- raw attractor countを一致させたrobust repertoireの独立30 seed確認
- robust repertoireと外乱下符号記憶を結ぶ事前登録30 seed確認
- 対称・疎・非対称・非正規4 familyの事前登録外的妥当性確認
- 結合norm、局所Jacobian、固定点座標、非正規性の比較baseline
- 出典、実装、反証条件を接続する研究主張台帳

これらは基礎指標であり、全アトラクタの発見やESPの数学的証明を保証しない。
適用条件は [基礎指標の数学的根拠](docs/research/theory/core-metrics.md) を参照する。

## 研究文書

- [研究計画](research.md)
- [証拠の運用規約](docs/research/README.md)
- [未解決課題](docs/research/open-questions.md)
- [継続文献調査](docs/research/literature-watch.md)
- [実験記録テンプレート](docs/research/experiments/TEMPLATE.md)
- [EXP-2026-001](docs/research/experiments/EXP-2026-001.md)
- [EXP-2026-002](docs/research/experiments/EXP-2026-002.md)
- [EXP-2026-003](docs/research/experiments/EXP-2026-003.md)
- [EXP-2026-004](docs/research/experiments/EXP-2026-004.md)
- [EXP-2026-005](docs/research/experiments/EXP-2026-005.md)
- [EXP-2026-006](docs/research/experiments/EXP-2026-006.md)
- [EXP-2026-007](docs/research/experiments/EXP-2026-007.md)
- [EXP-2026-008](docs/research/experiments/EXP-2026-008.md)
- [EXP-2026-009](docs/research/experiments/EXP-2026-009.md)
- [EXP-2026-010](docs/research/experiments/EXP-2026-010.md)
- [日本語論文草稿](docs/papers/robust-repertoire-memory-ja.md)
- [日本語論文DOCX](docs/papers/robust-repertoire-memory-ja.docx)
- [core–reserve保護条件](docs/research/theory/core-reserve-protection.md)
- [双安定coreのロバスト不変margin](docs/research/theory/bistable-core-margin.md)
- [非対角tanh RNNのorthant-box margin](docs/research/theory/orthant-box-margin.md)
- [生得的機能コアと可塑的力学余剰](docs/research/directions/innate-core-plastic-reserve.md)
- [論文化ゲート](docs/research/publication-readiness.md)

## Ground-truth実験

```bash
docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m reservoir_dynamics.experiments.scalar_tanh_contraction

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.input_conditioned_stability_sweep
docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.multidimensional_memory_sweep

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.replica_readout_validation

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.core_reserve_protection

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.bistable_core_protection

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m \
  reservoir_dynamics.experiments.orthant_count_confirmation
```

## テスト

Docker環境で実行する。

```bash
docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run --group dev coverage run -m unittest discover -s tests -v

docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run --group dev coverage report -m
```
