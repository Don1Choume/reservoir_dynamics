# 継続的な文献対応付け

最終確認: 2026-08-01

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
| topology外的妥当性 | symmetry, sparsity, modularity, task dependence | `H-RC-007`, `EXP-2026-010`, `EXP-2026-012` |
| 構造・機能・遺伝 | structural connectome genetics, eigenmodes, individual variability | `C-BIO-001`, `H-BIO-004`, `OQ-012` |
| 非正規・非相反力学 | non-normality, reactivity, transient amplification | `H-RC-007`, `EXP-2026-010` |

## 2026-07-30 差分

- Dhadphale et al. (2026) は、connectivity patternとedge weightを制御した
  5 reservoir topologyで、対称性の予測性能効果が対象力学系に依存することを
  報告した。DOI: <https://doi.org/10.1063/5.0314081>
- Poggialini et al. (2026, accepted) は、非相反Wilson–Cowan networkで
  feedforward構造が局所非正規性を固定しても過渡reactivityを増すことを
  報告した。DOI: <https://doi.org/10.1103/jv6l-3s5z>
- Clark (2026) は、連想記憶modelで固定点が消えた容量超過後にも有限時間の
  transient retrievalが残り得ることを示した。アトラクタcountと有限時間機能を
  同一視しない根拠になる。DOI: <https://doi.org/10.1103/42y2-bsh1>
- Park et al. (2026, preprint) は、photonic reservoirでsmall-world topologyの
  memory・予測性能を報告した一方、最適topologyがtaskに依存することを示した。
  arXiv: <https://arxiv.org/abs/2607.23285>

## 2026-08-01 差分

- Wainberg et al. (2024) は26,333人、206 tractography指標のGWASから、
  白質構造結合に広範で多遺伝子的、空間構造化された遺伝的影響を報告した。
  DOI: <https://doi.org/10.1038/s41467-024-46023-2>
- Sydnor et al. (2024) は4独立dataset、計3,355人の事前登録解析で、発達中の
  機能結合変化がsensorimotor-association軸に沿って再現することを示した。
  DOI: <https://doi.org/10.1038/s41467-024-47748-w>
- Xia et al. (2026) は白質、微細構造類似性、空間近接性を統合した多尺度
  structural connectome eigenmodeが安静時・task時活動を単一尺度近似より
  よく捉えることを報告した。
  DOI: <https://doi.org/10.1038/s42003-026-10558-5>
- Bian et al. (2024, preprint) はBaby Connectome Projectで個体差を保持する
  Bayesian module推定を提示した。発達共通性と個体差を分離する方法候補だが、
  遺伝・学習寄与の因果分解ではない。
  arXiv: <https://arxiv.org/abs/2407.13118>
- Raghav et al. (2026, preprint) は反復fMRIの測定誤差を明示したtwin modelで、
  遺伝・環境寄与が多尺度communityへ組織されることを報告した。査読前である。
  arXiv: <https://arxiv.org/abs/2604.24614>

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
