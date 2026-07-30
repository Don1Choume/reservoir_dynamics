# 継続的な文献対応付け

最終確認: 2026-07-30

## 監視対象

| 領域 | 検索軸 | 研究への接続 |
|---|---|---|
| Generalized RC | `"reservoir computing generalized"`, output reproducibility | `C-RC-003`, `H-RC-005` |
| 機能的整合性 | output/readout consistency, consistency profile, replica test | `C-RC-012`, `H-RC-006` |
| 条件付き安定性 | conditional Lyapunov, bubbling, generalized synchronization | `C-RC-001`, `H-RC-001` |
| 多機能・連想記憶 | multifunctional RC, associative memory, spurious attractor | `H-RC-002` |
| 遷移理論 | committor, MFPT, quasipotential, TPT, metastability | `H-RC-003` |
| 大域解析 | basin stability, Conley–Morse, transfer operator | `C-DYN-001` |
| 過渡安全性 | survivability, robust invariant set, ISS, tipping margin | `C-DYN-003`, `C-RC-015` |
| NN不変集合 | neural dynamical system, set recursion, reachability, hyperbox | `OQ-010` |
| robust repertoire | disturbance margin, viability kernel, safe basin, task robustness | `C-RC-019`, `H-RC-007` |
| 容量理論 | IPC, TIPC, fading memory, universality | `C-RC-002` |
| 調整 | differentiable dynamics, topology intervention, reservoir design | `H-RC-004` |
| 発生prior | genomic bottleneck, developmental encoding, innate circuit | `C-BIO-001`, `H-BIO-001` |
| 発生connectome | connectome maturation, pruning, CA3 development | `C-BIO-002`, `H-BIO-002` |
| 進化と生涯学習 | evolution-learning loop, plasticity mask, critical period | `H-BIO-001`, `H-BIO-003` |
| 力学的干渉保護 | dynamical subspace, orthogonal gradient, modular continual learning | `C-BIO-003`, `H-BIO-004`, `H-BIO-005` |
| motif再利用 | dynamical motif, compositional RNN, fast task transfer | `C-RC-017`, `H-BIO-002` |
| plasticity枯渇 | loss of plasticity, feature renewal, continual learning | `C-CL-001`, `H-BIO-002` |

## 情報源の優先順位

1. 査読誌の原著論文と出版社ページ
2. arXivと著者公開原稿
3. 公式ソフトウェア文書とrelease note
4. review、perspective
5. 二次記事は探索にのみ使い、主張の根拠にしない

## 更新時の手順

1. 公開日と実際の研究実施時期を分けて確認する。
2. abstractだけでなく、仮定、比較条件、supplementary materialを確認する。
3. 新規論文が既存主張を支持、限定、反証のどれに当たるか記録する。
4. `claims.toml` のsource追加だけで済ませず、`limitations` と反証条件を見直す。
5. 実装へ影響する場合は、先に回帰テストを追加する。

少なくとも各実験フェーズ開始時と論文投稿前に再検索する。急速に変化する
GRC、bubbling、アトラクタ設計については月次確認候補とする。
