# 継続的な文献対応付け

最終確認: 2026-07-30

## 監視対象

| 領域 | 検索軸 | 研究への接続 |
|---|---|---|
| Generalized RC | `"reservoir computing generalized"`, output reproducibility | `C-RC-003`, `H-RC-005` |
| 条件付き安定性 | conditional Lyapunov, bubbling, generalized synchronization | `C-RC-001`, `H-RC-001` |
| 多機能・連想記憶 | multifunctional RC, associative memory, spurious attractor | `H-RC-002` |
| 遷移理論 | committor, MFPT, quasipotential, TPT, metastability | `H-RC-003` |
| 大域解析 | basin stability, Conley–Morse, transfer operator | `C-DYN-001` |
| 容量理論 | IPC, TIPC, fading memory, universality | `C-RC-002` |
| 調整 | differentiable dynamics, topology intervention, reservoir design | `H-RC-004` |

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
