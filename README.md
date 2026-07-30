# Reservoir Dynamics

任意の力学系をリザバーとして分析・調整するための研究用ツールである。
自律、入力駆動、閉ループのダイナミクスを区別し、アトラクタ、吸引域、
遷移、条件付き安定性、情報処理容量を同一の証拠規約で扱う。

## 現在の実装範囲

- 吸引確率とWilson信頼区間によるbasin stability推定
- Shannon entropyに基づく実効レパートリー
- 同一入力を受ける複製軌道のpairwise RMS距離
- 共通入力を複数初期状態へ与える離散時間シミュレータ
- スカラーtanhリザバーの収縮上界とground-truth実験
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
- [論文化ゲート](docs/research/publication-readiness.md)

## Ground-truth実験

```bash
docker run --rm --gpus all -v ${PWD}:/app -w /app \
  -e PREFECT_API_URL=http://host.docker.internal:4200/api \
  -e PYTHONPATH=/app/src \
  sandbox-selenium \
  uv run python -m reservoir_dynamics.experiments.scalar_tanh_contraction
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
