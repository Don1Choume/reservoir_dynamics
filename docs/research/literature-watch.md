# 継続的な文献対応付け

最終確認: 2026-08-02

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
| glia・細胞外イオン | astrocyte calcium/sodium/chloride, extracellular ion, K buffering | `H-BIO-005`, `OQ-014` |
| 空間neuromodulation | dopamine/ACh wave, volume transmission, receptor map, reaction diffusion | `H-BIO-005`, `OQ-014` |
| 場によるreservoir制御 | spatial gate, local control, low-rank control, energy-matched intervention | `C-RC-029`, `EXP-2026-015` |
| 成分合成と規模外挿 | small-gain, compositional certificate, subsystem generalization, unknown partition | `C-RC-030`, `H-RC-008`, `EXP-2026-016`, `EXP-2026-017`, `OQ-012` |

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

## 2026-08-02 差分

- Cahill et al. (2024) は局所的で短いGABA・glutamate入力が、より広く長い
  astrocyte Ca network応答へ変換されることを示した。
  DOI: <https://doi.org/10.1038/s41586-024-07311-5>
- 海馬astrocyteの求心的時系列統合がlocus coeruleusにより調整されること、
  striatumで局所細胞外Caがcholinergic interneuronとdopamine放出をsubsecondで
  変えることが報告された。
  DOI: <https://doi.org/10.1038/s41593-024-01612-8>,
  <https://doi.org/10.1038/s41467-024-54253-7>
- astrocyte Na恒常性の細胞・細胞内不均一性とK取り込み、brain state依存の
  astrocyte Clが報告され、単一一様scalarではない局所場の候補を補強した。
  DOI: <https://doi.org/10.1038/s41467-026-73435-z>,
  <https://doi.org/10.1038/s41467-023-37433-9>
- ACh–dopamineの時空間waveと、dopamineがplasticity・excitabilityを介して
  latent behavioral attractorを形成・顕在化させる例を確認した。
  DOI: <https://doi.org/10.1038/s41467-023-42311-5>,
  <https://doi.org/10.1038/s41467-024-53976-x>
- 以上は時空間的に不均一な調整の存在を支持するが、core–reserve構造や
  `EXP-2026-015` の拡散場方程式を直接支持する証拠ではない。
- Dashkovskiy, Rüffer, and Wirth (2010) は、相互結合した非線形subsystemの
  ISS Lyapunov関数をgain operatorのsmall-gain条件で全体系へ構成する枠組みを
  与えた。方向別結合負荷を保持する理論的根拠であるが、本研究のhyperbox
  certificateやtask保持率積則そのものの証明ではない。
  DOI: <https://doi.org/10.1137/090746483>
- Zhang et al. (2023) は小規模subsystemで得たISS neural certificateを類似構造の
  大規模networkへ合成・一般化する方法を提示した。`EXP-2026-016` の小系から
  大系へのcertificate feature移送と問題設定を共有するが、対象taskと保証形式は
  異なる。<https://proceedings.mlr.press/v211/zhang23a.html>
- Driscoll, Shenoy, and Sussillo (2024) はmulti-task RNNでattractor、decision
  boundary、rotation等のdynamical motifがtask間で再利用されることを示した。
  component atlasの生物・計算論的動機になるが、module独立性や積分解は示さない。
  DOI: <https://doi.org/10.1038/s41593-024-01668-6>
- Pradhan, Dasgupta, and Sinha (2011) はbinary attractor networkでattractor basin
  と収束時間が中間的なmodularityで改善し得ることを報告した。module化は常に
  単調に有利ではなく、結合強度・task・basinを同時に測る必要がある。
  DOI: <https://doi.org/10.1209/0295-5075/94/38004>
- これらの文献はcomponent-aware分析の方向を支持するが、2+2・2+3から3+5への
  MAE改善は本研究固有の経験的結果である。別generator、未知分割、3 module以上、
  stochastic taskへの一般化は未検証として維持する。
- Zhou et al. (2025, preprint) は、離散時間の相互結合系について規模に依存しない
  input-to-state stability certificateの合成・検証を提案した。大規模化で局所
  certificateを再利用する比較対象だが、査読前であり、attractor repertoireや
  module推定を扱わない。<https://arxiv.org/abs/2509.10118>
- Diez et al. (2026) はhuman connectomeで、早い神経発生時期と構造中心性、
  発生時期が近い領域間の結合確率・重み、発生関連遺伝子発現の関連を報告した。
  発生blueprintの構造priorを支持するが、力学moduleまたはアトラクタを同定した
  研究ではない。DOI: <https://doi.org/10.1038/s41467-025-67785-3>
- Nano et al. (2025) はhuman cortexの発生・成人single-cell atlasを統合し、
  細胞subtype指定に対応する時空間的gene co-expression meta-moduleを同定した。
  細胞型生成規則の根拠であり、計算moduleや安全marginの証拠ではない。
  DOI: <https://doi.org/10.1038/s41593-025-01933-2>

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
