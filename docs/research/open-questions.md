# 未解決課題と識別実験

最終確認: 2026-07-30

## 優先度A: 理論と最初の実験を直結する課題

### OQ-001 自律アトラクタは入力駆動能力をどこまで予測するか

- 対応仮説: `H-RC-001`
- 未知部分: 自律系の多重安定性と入力条件付き応答の関係は、入力gainや
  generalized synchronizationにより変わる。
- 識別実験: 同一RNN個体について、自律atlas、replica距離、条件付き
  Lyapunov指数、IPC/TIPC、未知task性能を同時測定する。
- 必要な対照: 状態次元、重み分布、読み出し、探索予算を一致させる。
- 実装済み: 共通入力を複数初期状態へ配信する離散時間replica simulator。
- 検証済み: `EXP-2026-002` で、負の局所条件付き指数と非同期replicaが
  共存する場合、および大域収縮条件外の入力同期を再現した。
- 検証済み: `EXP-2026-003` で、多次元tanh RNNでも負のtop conditional
  Lyapunov指数とreplica非同期を分離し、linear memory curveを同時測定した。
- 検証済み: `EXP-2026-004` の30 seedで、強入力による同期・shared worst
  容量の回復と、強収縮条件の局所線形記憶優位を区間推定付きで再現した。
- 陰性結果: 負CLE・非同期率0.8以上、および局所容量とshared worst容量の
  絶対差の95%区間下限0.5超という事前登録基準は不成立だった。
- 探索的所見: 状態同期率0.7333でも固定readout retentionが0.9962の条件と、
  同期率0でretentionが0.0892の条件を分離した。
- 次の識別実験: 自律atlasとtask-specific機能同値類を同時推定し、
  Lymburn型output consistencyに対する未知初期状態性能の増分説明率を測る。

### OQ-002 実効レパートリーは連想記憶容量の媒介変数か

- 対応仮説: `H-RC-002`
- 未知部分: \(N_{\mathrm{eff}}\) は多様性を表すが、識別性と偽アトラクタを
  含まない。
- 識別実験: 生の個数を固定し、吸引域の偏りだけを変える介入を行う。
- 主要評価: cue corruption別想起率、保持曲線AUC、偽想起率。
- 次の実装: 軌道signature分類器と未分類labelを保持するbasin sampler。

### OQ-003 遷移時間とタスク遅延の整合

- 対応仮説: `H-RC-003`
- 未知部分: 有益なのがMFPT、mixing time、implied timescaleのどれかは
  決まっていない。
- 識別実験: 力学系側の障壁とタスク側の必要遅延を独立に掃引する。
- 次の実装: 状態遷移count、lag-time検証、MFPTの推定器。

## 優先度B: 調整と一般化

### OQ-004 微分可能proxyは真のアトラクタ特性を改善するか

- 対応仮説: `H-RC-004`
- 危険: 有限時間proxyや積分器の数値誤差だけを最適化する可能性がある。
- 識別実験: solver、時間刻み、観測時間を変えた非微分認証をblindで行う。

### OQ-005 GRCの能力はどこに宿るか

- 対応仮説: `C-RC-003`, `H-RC-005`
- 未知部分: リザバー、履歴表現、非線形読み出し間の仕事量分担。
- 識別実験: reservoir shuffle、reservoir除去、履歴長、読み出し幅を
  factorial designで操作する。
- 主要評価: 状態と出力それぞれのreplica距離、データ量、FLOPs、未知初期値性能。

### OQ-009 task-specificな機能的アトラクタ商は有効か

- 対応仮説: `H-RC-006`
- 未知部分: raw stateでは異なる応答を、どのreadout、task集合、許容誤差で
  同じ機能としてまとめるべきか。
- 現在の代理量: 参照replicaだけで学習した固定readoutのheld-out \(R^2\) と
  shared worst retention。
- 既存研究との差: global/output consistencyの分離は既知である。本研究では
  atlas上の各アトラクタをtask別の同値類へまとめ、利用可能容量と調整目標へ
  接続できるかを問う。
- 識別実験: 線形、二次、履歴readoutを公平な複雑度で比較し、未知初期状態と
  未知taskに対する増分予測力を交差検証する。
- 数学課題: score差による近似関係は推移的とは限らない。厳密な出力同値関係、
  擬距離、安定なclustering規則を区別する。

## 優先度C: 生得的機能コアと生涯学習

### OQ-006 力学的余剰をどう定義するか

- 対応仮説: `H-BIO-001`, `H-BIO-002`
- 未知部分: 未使用アトラクタ個数、可塑parameter数、アクセス可能次元、
  catastrophic forgetting耐性のどれが本質的か。
- 操作定義: core性能低下を \(\varepsilon\) 以下に制約したとき、学習budget
  \(B\) で達成できる未知task改善量をplastic reserveとする。
- 検証済み: `EXP-2026-005` のblock-triangular構成では、reserve-only更新が
  core retention 1を保ちながらnovel線形記憶容量を平均3.5503追加した。
- 検証済み: `EXP-2026-006` の双安定scalar coreでは、学習前に不活性な
  reserveへ正負二つのcue保持アトラクタを形成し、臨界feedback比0.9まで
  認証区間のcore retention 1を保った。
- 証拠境界: これは収縮・対角またはscalar双安定tanh系の構成的十分性であり、
  発生過程、高次元学習、random mask比較は未検証である。
- 識別実験: 静的アトラクタ数を一致させ、plasticity maskとmodule境界だけを
  変える。
- 主要評価: core保持、新規task sample efficiency、アトラクタ新生・消失、
  replica同期、energy。
- 詳細:
  [生得的機能コアと可塑的力学余剰](directions/innate-core-plastic-reserve.md)

### OQ-007 発生generatorは何を符号化すべきか

- 対応仮説: `C-BIO-001`, `C-BIO-002`, `H-BIO-001`
- 候補: topology、細胞型、局所結線規則、初期重み、plasticity rule、
  critical period、neuromodulation。
- 識別実験: 同じgenotype記述長で符号化対象だけを変え、outer evolutionと
  inner lifetime learningの一般化を比較する。
- 必要な対照: test taskをouter loopへ漏洩させず、進化的overfittingを測る。

### OQ-008 人間規模の必要条件をどう下界化するか

- 対応仮説: `H-BIO-003`
- 未知部分: neuron数やsynapse数だけでは、task複雑度、時間尺度、energy、
  頑健性、生涯学習を表せない。
- 第一段階: 32から8192状態まで、task組合せ複雑度、アクセス可能次元、
  plastic reserve、energyのscale lawを推定する。
- 初期certificate: 収縮core modelでは、複数reserveからcoreへ入るfeedback
  budget \(\sum_j L_{f,j}\overline R_j\) を
  \((1-L_c)\varepsilon\) 以下に保てばworst-case偏差を制限できる。
- 多重安定certificate: 必須安全集合 \(S_k\) が外力norm \(\mu_k\) まで
  ロバストなら、
  \(\sum_j\|G_j\|R_j\le\min_k\mu_k\) で全安全集合を保護できる。
- 注意: 上記は十分条件であり、人間規模の必要条件または生物学的実測則では
  ない。
- 外部妥当性: network familyを跨ぐ予測と発生connectomeへの適合を要求する。
- 禁止事項: 外挿誤差を検証する前に、人間相当の単一数値を宣言しない。

### OQ-010 多重安定coreを保つcoupling marginは何か

- 対応仮説: `H-BIO-004`, `H-BIO-005`
- 部分解決: `EXP-2026-006` のscalar tanh双安定coreでは、
  \(m_*=\sqrt{1-1/a}\) と
  \(\eta_{\mathrm{crit}}=am_*-\operatorname{atanh}(m_*)\) を導き、
  認証区間の任意時変有界外力に対する保持を証明した。
- 再現結果: 臨界比0.5と0.9でcertified retention 1、1.1と1.5で反対cue
  retention 0。無外力basin全体の保持率は0.9341と0.8693に低下した。
- 未知部分: 高次元・非対角・非normal coreでは、安全集合、方向依存margin、
  複数basin境界を誘導normだけから決められない。
- 候補量: 固定点の局所収縮率、basin境界までの距離、minimum action、
  cue-to-attractor routing margin。
- 次の識別実験: 低次元非対角RNNでmaximal robust invariant set近似と
  Monte Carlo survivabilityを同時測定し、scalar bound、局所Jacobian、
  basin距離の予測誤差を比較する。
- 実装候補: Li et al. (2025) のNN dynamical system向けhyperbox集合再帰
  <https://arxiv.org/abs/2505.11546>を査読前baselineとして再現し、
  uncontrolled bounded disturbanceへ拡張できるか確認する。

### OQ-011 robust repertoire curveはtask性能を予測するか

- 対応仮説: `H-RC-007`
- 確認済み: `EXP-2026-008` の未使用30 seedではraw autonomous countを
  16対16に一致させ、orthant-box認証countを16.0対10.2へ分離した。
- 確認済み: `EXP-2026-009` の未使用30 seed、raw count一定120条件では、
  低外乱のcertified fractionと符号記憶保持率のSpearmanが0.8823、高外乱の
  mean marginとのSpearmanが0.9347だった。raw-count予測よりMAEが改善した。
- 確認済み: `EXP-2026-010` の未使用30 seed、4 family、raw count一定
  480条件ではmean marginとのSpearmanが0.8933–0.9771だった。seed単位pooled
  MAEはraw count、coupling、worst local Jacobianよりそれぞれ0.0851、
  0.0080、0.0051小さかった。
- 未知部分: hyperbox certificateの順位がfamilyを丸ごと未知にした予測、
  stochastic noise、cue、readout taskでも保たれるか未確認である。
- 次の識別実験: leave-one-family-out予測と、time-varying stochastic外乱に
  対するsurvival curveを行う。続いてcue routingとlinear readout保持率を測る。
- baseline: raw count、平均局所Jacobian spectral radius、\(\|W\|_\infty\)、
  basin stability、minimum fixed-point coordinate。
- 主要判定: 未知family・未知外乱時系列で、\(N_{\mathrm{rob}}(e)\) または
  \(S_{\mathrm{rob}}(e)\) がcoupling、局所安定性、多変量baselineへ増分予測力を
  持つ。
- 外的妥当性: 高次元、学習済み、spiking、物理reservoirへ拡張する。
- 主要評価: core attractor survival、basin stability差、機能的商の保持、
  novel容量、energy。

## 運用規則

- 各課題は対応する主張IDを持たせる。
- 主要評価量と反証条件を決める前に大規模計算を始めない。
- 結果が陰性でも課題を削除せず、`claims.toml` を `refuted` へ更新する。
- 新しい数学的主張には、仮定、定理または命題、証明、数値上の代理量を分けて記す。
