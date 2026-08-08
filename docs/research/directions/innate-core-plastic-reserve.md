# 研究方向: 生得的機能コアと可塑的力学余剰

最終確認: 2026-08-02
状態: 長期仮説。生物学的着想には支持証拠があるが、アトラクタ余剰としての
定式化は未検証。

## 1. 中心仮説

進化は神経回路の全結合を直接指定するのではなく、発生規則、細胞型、
結線prior、可塑性規則を圧縮して遺伝的に指定する。この発生プログラムは
生存に必要な機能コアを形成すると同時に、生涯学習で新しい機能を獲得できる
力学的余地を残す。

本研究ではこれを次の仮説へ分解する。

- `H-BIO-001`: 圧縮された発生規則から生成される機能コアは、完全random初期化
  より少ない生涯学習標本で新しい課題へ適応できる。
- `H-BIO-002`: 生得的性能を最大化し過ぎると可塑性が失われ、可塑性を最大化
  し過ぎると初期機能と頑健性が失われる。最適点は両者のPareto frontier上にある。
- `H-BIO-003`: 有用な「余剰」は未使用アトラクタの静的個数ではなく、既存機能を
  保持したまま新しい安定応答を形成できる反実仮想的な適応容量である。
- `H-BIO-004`: 発生中の構造化、疎化、modularity、可塑性maskは、同じ可塑
  parameter数の一様なネットワークより干渉を減らす。
- `H-BIO-005`: coreへ到達する学習・活動負荷を、局所受容体、抑制、
  neuromodulator、glia・細胞外イオン場に相当する空間gateでmargin内へ制御
  できれば、一様な全体制御よりcore保持とreserve利用を両立しやすい。

## 2. 既存研究との対応

### 出典付き事実

Zadorは、脳の結線全体をゲノムが個別指定することはできず、発生規則として
圧縮されるという`genomic bottleneck`を提示した。

- [Zador 2019](https://doi.org/10.1038/s41467-019-11786-6)

発生規則を外側の進化loop、個体内学習を内側のloopとして扱う計算研究では、
重み行列より低次元な結線規則がtask-relevantな回路を生成し、圧縮と適応性を
両立できる例が示されている。

- [Barabási et al. 2023](https://doi.org/10.1038/s41467-023-37980-1)
- [Shuvaev et al. 2024](https://doi.org/10.1073/pnas.2409160121)

生物学的にも、発生初期から保存される構造的scaffoldと、その上で変化する
結合が共存する。isogenicな線虫でも個体固有の結合差があり、発生に伴って
synapse数と結合強度が選択的に変化する。

- [Witvliet et al. 2021](https://doi.org/10.1038/s41586-021-03778-8)

幼生zebrafishでは、発生中のspiking activityを抑制しても視覚運動回路の
主要機能が形成された例がある。ただし、これは全脳機能一般が経験非依存である
ことを意味せず、当該動物・回路・課題に限定された結果である。

- [Barabási et al. 2024](https://doi.org/10.1038/s41467-023-44681-2)

一方、2026年のmouse CA3研究では、発生に伴い局所的で密なrandom結合から、
疎で広域かつ構造化された自己連想回路へ変化し、model上の記憶性能も改善した。
これは発生後の経験依存的構造化と整合するが、遺伝と経験の寄与率を単独で
決定したものではない。

- [Vargas-Barroso et al. 2026](https://doi.org/10.1038/s41467-026-71914-x)

ヒト構造connectomeでは、皮質領域の発生時期が構造中心性と相関し、神経新生時期が
近い領域ほど結合確率と結合重みが大きいという一次解析が報告された。またヒト皮質の
発生transcriptomeでは、細胞subtype指定に関係する500超のgene co-expression
networkとmeta-moduleが同定された。これらは発生時期と分子programが成人構造の
scaffoldへ制約を与えることと整合するが、力学的attractor module、機能core、
plastic reserveの存在を直接示す証拠ではない。

- [Diez et al. 2026](https://doi.org/10.1038/s41467-025-67785-3)
- [Nano et al. 2025](https://doi.org/10.1038/s41593-025-01933-2)

multitask人工RNNでは、attractor、decision boundary、rotationなどの
dynamical motifがtask間で再利用され、既存motifを入力で再構成することで
新taskを速く学べる例が示された。これは、学習余剰が空のparameterや未使用
attractorだけでなく、再利用可能な既存力学部品にも宿ることを示す。

- [Driscoll et al. 2024](https://doi.org/10.1038/s41593-024-01668-6)

また、標準的なdeep networkは継続的な非定常task列でplasticity自体を失い、
低utility featureを継続的に置換する方法がplasticityを維持した。これは
attractor研究ではないが、「余剰」は初期量だけでなく、利用・再生・枯渇を
追跡すべき動的資源であることを示す比較対象になる。

- [Dohare et al. 2024](https://doi.org/10.1038/s41586-024-07711-7)

神経回路の調整はsynapse重みだけではない。局所的で短いGABA・glutamate入力が
より広く分単位のastrocyte Ca応答へ統合されること、locus coeruleus入力が
海馬astrocyteの秒単位の求心的統合を調整すること、細胞外Caの局所変化が
subsecondでstriatal cholinergic interneuronとdopamine放出を変えることが
報告されている。

- [Cahill et al. 2024](https://doi.org/10.1038/s41586-024-07311-5)
- [Centripetal integration in hippocampal astrocytes 2024](https://doi.org/10.1038/s41593-024-01612-8)
- [Rapid astrocyte modulation of extracellular Ca 2024](https://doi.org/10.1038/s41467-024-54253-7)

さらにastrocyteのNa恒常性は細胞内・細胞間で不均一でK取り込みと結び付き、
astrocyte Cl濃度もbrain stateに依存する。AChとdopamineの放出は局所releaseと
reaction–diffusionで記述される時空間waveを形成し得る。またdopamineが
plasticityとexcitabilityを介して潜在的な行動attractorを形成・顕在化させる
例もある。

- [Astrocytic sodium homeostasis 2026](https://doi.org/10.1038/s41467-026-73435-z)
- [Brain-state-dependent astrocytic chloride 2023](https://doi.org/10.1038/s41467-023-37433-9)
- [Acetylcholine–dopamine waves 2023](https://doi.org/10.1038/s41467-023-42311-5)
- [Dopamine and latent behavioral attractors 2024](https://doi.org/10.1038/s41467-024-53976-x)

これらは、調整信号が空間的に一様なscalarではなく、複数の時定数と局所性を
持ち得ることを支持する。ただしcore–reserve分解、以下の拡散方程式、または
局所gateの最適性を生物学的に示したものではない。

### 本研究の推論

以上は「固定された生得回路」と「自由な後天学習」の二分法より、

1. 圧縮された発生generator
2. 生成された初期回路
3. 空間・時期・細胞型ごとに異なる可塑性規則
4. 経験で形成される条件付きアトラクタ

の階層で捉えるべきことを示唆する。

ただし、既存研究は「未使用アトラクタ数」を直接測っていない。この部分は
本研究独自の仮説であり、既存事実として扱わない。

## 3. 数学的な操作定義

遺伝的に伝達される低次元記述を \(g\)、発生noiseと環境を
\(\xi_{\mathrm{dev}}\) とし、発生写像を

\[
\mathcal{D}(g,\xi_{\mathrm{dev}})
\longmapsto
(W_0,P,M)
\]

とする。

- \(W_0\): 学習前の回路
- \(P\): 局所可塑性則とそのhyperparameter
- \(M\): 可塑性を許す結合、時期、moduleのmask

生涯経験 \(E\) による学習写像を

\[
W_E=\mathcal{L}(W_0,P,M,E)
\]

とする。

### 機能コア

学習前または極少数標本で要求されるtask集合
\(\mathcal{T}_{\mathrm{core}}\) に対する性能を

\[
F_{\mathrm{core}}(W)
=
\mathbb{E}_{\tau\sim\mathcal{T}_{\mathrm{core}}}
[S_\tau(W)]
\]

と定義する。反射、恒常性、基本的sensorimotor変換などに相当するが、
人工実験では明示したtask suiteへ置き換える。

### 可塑的余剰

静的なアトラクタ個数ではなく、許容学習予算 \(B\) とcore性能の許容低下
\(\varepsilon\) の下で達成できる新規task改善量として

\[
R(B,\varepsilon)
=
\sup_{\Delta W\in\mathcal{A}(P,M,B)}
\left[
F_{\mathrm{new}}(W_0+\Delta W)-F_{\mathrm{new}}(W_0)
\right]
\]

subject to

\[
F_{\mathrm{core}}(W_0+\Delta W)
\ge
F_{\mathrm{core}}(W_0)-\varepsilon
\]

および安定性、エネルギー、配線量の制約、と定義する。

この定義では、余剰は次の要素を含む。

- 新しい安定応答またはmetastable sequenceを形成する能力
- 既存のdynamical motifを新しいtaskへ再構成・再利用する能力
- 既存吸引域を壊さずにdecision boundaryを追加する能力
- 入力条件付き表現次元と読み出し可能性を増やす能力
- catastrophic forgettingを抑える分離性
- 使用済み自由度を再生し、将来taskへのplasticityを維持する能力

アトラクタ数は説明変数の一つだが、余剰そのものではない。

従ってplastic reserveを少なくとも、

\[
R_{\mathrm{total}}
=
R_{\mathrm{novel}}
+
R_{\mathrm{reuse}}
+
R_{\mathrm{renew}}
\]

という概念上の三成分へ分ける。

- \(R_{\mathrm{novel}}\): 新しい機能同値類を形成する容量
- \(R_{\mathrm{reuse}}\): 既存motifを新しいroutingで再構成する容量
- \(R_{\mathrm{renew}}\): plasticity loss後も自由度を再生する容量

この和は現段階で加法的な実測式ではない。成分が重複する可能性があるため、
介入で分離可能かを今後検証するための概念分解である。

さらに、raw state上で異なるアトラクタが、固定readoutとtaskから見ると同じ
機能を実現する場合がある。従って、plastic reserveを数える単位は、常に
物理的なアトラクタ一個であるとは限らない。readout \(h\) と評価時間集合
\(\mathcal I\) に対する厳密な出力同値関係を明示し、

\[
\mathcal A_{\mathrm{functional}}
=
\mathcal A_{\mathrm{state}}/{\equiv_{h,\mathcal I}}
\]

というtask-specificな機能的商を候補とする。学習による有用な余剰は、この
商空間で新しい機能同値類を形成する能力、または既存類の頑健な吸引域を拡張
する能力として測る。単なるstate attractorの分裂がreadout出力を変えない
なら、機能余剰の増加とは数えない。

実験では出力擬距離の有限閾値とclusteringで近似する。この商構造は現段階の
仮説である。task集合間の整合性、閾値感度、有限標本での推定誤差は、
`H-RC-006` と `OQ-009` で検証する。

### 最初の保護certificate

`EXP-2026-005` では、coreがreserveからfeedbackを受けないblock-triangular
tanh系を構成し、reserve-only更新がcore軌道と固定core readoutを厳密に
保存することを証明・再現した。有限feedback \(L_f\) がある場合も、

\[
D_t
\le
L_c^tD_0
+
L_f\overline R
\frac{1-L_c^t}{1-L_c}
\]

でcore偏差を上から抑えられる。

これはplastic reserveの存在可能性を示す構成的十分条件であり、脳内の実際の
module境界を示す証拠ではない。詳細は
[機能coreと可塑reserveの保護条件](../theory/core-reserve-protection.md)
に記録する。

### 多重安定coreの保護certificate

`EXP-2026-006` では、双安定scalar tanh coreへ拡張した。coreを

\[
c_{t+1}=\tanh(ac_t+\eta_t),\qquad a>1
\]

とすると、

\[
m_*=\sqrt{1-\frac1a},
\qquad
\eta_{\mathrm{crit}}
=
am_*-\operatorname{atanh}(m_*)
\]

であり、\(|\eta_t|\le\eta_{\mathrm{crit}}\) なら
\([m_*,1]\) と \([-1,-m_*]\) は任意の時変外力に対してロバスト正不変に
なる。

学習前に不活性だったreserveへcue依存の正負二つの自律固定点を形成した
30 seed実験では、臨界比0.5と0.9で認証区間のcore retentionが1、臨界比
1.1と1.5では反対cueの保持率が0になった。

重要なのは、無外力の同じbasin全体から始めると、臨界比0.5と0.9でも保持率が
0.9341と0.8693に低下したことである。アトラクタの存在またはbasin所属だけ
では、学習外乱に対する機能保持を保証しない。

詳細は
[双安定coreのロバスト不変margin](../theory/bistable-core-margin.md)と
[EXP-2026-006](../experiments/EXP-2026-006.md)に記録する。

### 空間変調場による座標別保護

`EXP-2026-015` では変調場を

\[
z_{t+1}=(1-\alpha-\beta)z_t+\alpha Pz_t+\beta s_t,
\qquad g_t=1-z_t
\]

とし、row-stochasticな \(P\)、\(\alpha,\beta\ge0\)、
\(\alpha+\beta\le1\)、\(z_0,s_t\in[0,1]^n\) の下で
\(z_t\in[0,1]^n\) が不変であることを示した。reserveからcoreへの行列の
各受信座標を \(g_t\) でgateし、同じFrobenius介入energyを持つ一様global
recurrent gainを対照とした。

3+5 node、非対称かつ双方向bridge、3 feedback gain、3 stochastic noise、
未使用30 seedの270条件では、局所gateの8必須core orthant保持率が全条件1だった。
global対照との差は0.3472（95% bootstrap区間 [0.2393, 0.4681]）、reserve
線形記憶容量差は0.07683（[0.06350, 0.09123]）で、2,851,200座標遷移に
certificate違反はなかった。これは空間局所性の構成的十分例であり、特定の
生体物質による実装証明ではない。理論と事前登録は
[空間変調場](../theory/spatial-modulation-fields.md)および
[EXP-2026-015](../experiments/EXP-2026-015.md)に記録する。

### 成分別certificateと規模外挿

二つのmoduleを

\[
x^+=\tanh(Ax+By+\eta_x),\qquad
y^+=\tanh(Cx+Dy+\eta_y)
\]

とし、隔離時の座標別marginを \(M_x,M_y\)、方向別結合負荷を
\(L_x=\|B\|_\infty\)、\(L_y=\|C\|_\infty\) とする。
外乱budget \(e\) に対する十分条件は

\[
M_x\ge e+L_x,\qquad M_y\ge e+L_y
\]

となる。実際のcross-edgeを座標別に移送したrectangle、方向別norm、全結合を
一つへ潰したglobal normの認証率を
\(R_{\mathrm{rect}},R_{\mathrm{dir}},R_{\mathrm{global}}\) とすれば、
実装したhyperbox familyでは

\[
T_{\mathrm{coupled}}\ge R_{\mathrm{rect}}
\ge R_{\mathrm{dir}}\ge R_{\mathrm{global}}
\]

が成り立つ。この順序はcomponent間の向きと局所marginを残すほど保守性が緩む
ことを表すが、一般の非矩形安全集合に自動的に拡張される定理ではない。

`EXP-2026-016` では2+2・2+3系だけでfitした方向別component predictorを、再fit
せず3+5系へ適用した。960点でMAEは0.01232となり、global profileの0.03970、
isolated task積だけの0.01699を上回った。全983,040 challengeで上記chainとtask
下界に違反はなかった。一方、Spearmanは0.8116で二baselineより低かったため、
これは絶対値較正の支持であり、順位支配や任意規模へのscale lawの立証ではない。
詳細は[方向別component結合](../theory/directional-component-coupling.md)と
[EXP-2026-016](../experiments/EXP-2026-016.md)に記録する。

### 未知三module分割と因子化合成

任意個のmodule partition \(I_1,\ldots,I_m\) に対し、受信方向別loadを

\[
L_{ij}=\max_{r\in I_i}\sum_{c\in I_j}|W_{rc}|,
\qquad \ell_i=\sum_{j\ne i}L_{ij}
\]

と定義する。module \(i\) の局所margin profileから外乱 \(e+\ell_i\) を
耐える割合を \(r_i(e+\ell_i)\) とすれば、方向別因子化certificateは

\[
R_{\mathrm{dir}}(e)=\prod_{i=1}^{m}r_i(e+\ell_i)
\]

で得られる。局所次元を \(d_i\)、最大局所次元を \(b\) とすると、全系の
\(2^{\sum_i d_i}\) orthantを列挙せず、\(\sum_i2^{d_i}\le m2^b\) の局所列挙で
計算できる。これは各moduleの局所安全性を合成する構成的十分条件であり、
一般の非矩形安全集合または誤分割に対する保証ではない。

`EXP-2026-017` では一意なinter/intra affinity gapを持つ座標permutation済み
2+2+3 tanh RNNについて、task前に240/240 partitionを完全回復した。局所16
orthantからの因子化値は全系128 orthantの直積列挙と960点すべてで一致した。
EXP-2026-016で固定した二module用predictorを再fitせず適用したMAEは0.01626で、
global 0.07993、product-only 0.02102より小さかった。一方、globalのSpearmanが
0.9468で最も高く、全moduleの最大流入loadを等しくした設計ではdirectionalと
global certificateも一致した。従って、本結果は絶対較正と計算量分解の支持であり、
順位支配、一般community回復、方向別loadの利得、人間規模則の証明ではない。
詳細は[多成分合成](../theory/multicomponent-composition.md)と
[EXP-2026-017](../experiments/EXP-2026-017.md)に記録する。

### 分割の摂動保証半径

`EXP-2026-018` では、最大affinity gapを \(g_1\)、二番目を \(g_2\) として

\[
r_{part}=\min\left(\frac{g_1}{2},\frac{g_1-g_2}{4}\right)
\]

を導出した。entrywise重み誤差がこの半径未満なら、pair affinityのthreshold前後と
最大gapの選択が同時に保たれるため、推定partitionも不変である。未使用30 seed、
180 base network、10,080摂動条件の事前登録確認では、半径0.9倍以下の5,760条件で
回復率1、pair disagreement 0だった。半径2倍では回復率0.7375、4倍では0.1653へ
低下した。

この量は「生得的scaffoldが個体差またはmodel誤差をどこまで許容できるか」の
人工系における候補指標である。ただし同じ構造partitionを保つだけで、同じ機能、
attractor、task、発生経路を保つとは限らない。詳細は
[EXP-2026-018](../experiments/EXP-2026-018.md)に記録する。

## 4. 反証可能な予測

### 予測P1: 逆U字型の適応frontier

core固定度が低過ぎれば初期性能とnoise頑健性が悪化し、高過ぎれば新規task
学習と分布shift適応が悪化する。

反証条件:

- core固定度を独立に操作しても、初期性能と適応性能のtrade-offが現れない。
- 一方の極端が全task family、全budgetで同時に優越する。

### 予測P2: 構造化された可塑性mask

同じ可塑parameter数、同じ更新回数、同じenergy budgetなら、module境界に
沿ったplastic reserveは一様random maskよりcore保持率を高める。

`EXP-2026-005` では、同じ8可塑parameter枠、同じ6候補、同じcalibration
系列で、reserve-only更新がcore更新対照よりnovel容量とcore retentionを
ともに改善した。ただしrandom matched maskとはまだ比較していないため、
予測P2全体の検証ではなく、極端なblock構成の支持例である。

反証条件:

- 未知taskと未知seedで一様maskと差がない、または一貫して劣る。

### 予測P3: 局所容量と大域利用可能性の分離

一つの吸引域内で高いmemory capacityを持つ回路でも、replica同期または
cue-to-attractor routingが不十分なら、個体全体の再現可能性能は低い。

`EXP-2026-004` は、この予測をより細かくした。状態非同期を含んでも固定
readout retentionが0.9962の条件がある一方、別の非同期条件では0.0892まで
低下した。従って、予測P3は「状態非同期なら性能が低い」ではなく、
「初期状態差がtask-relevantなreadout方向へ投影されると性能が低い」と修正
する。これは30 seedの単一RNN familyにおける探索的所見であり、生物学的な
plastic reserveを直接支持する証拠ではない。

### 予測P4: 発生scheduleの効果

初期の密な探索的結合から、coreを保存しつつ疎で構造化された回路へ移行する
scheduleは、最初から同じ最終疎度を持つ回路より学習効率と頑健性を高める。

### 予測P5: 必須アトラクタのmargin budget

発生または初期学習で形成された必須アトラクタには、task-relevantな安全集合と
外乱marginがある。新規学習moduleからのfeedback loadが最小marginへ近づくほど、
既存機能の誤遷移率と忘却率が増える。gating、抑制、疎結合、休眠moduleの
活性化は、このloadをtaskと時間に応じて制御する。

反証条件:

- core安全marginとfeedback loadを独立に操作しても、既存機能保持に
  再現可能な境界または単調関係がない。
- 複数の高次元network familyで、marginを含まない結合強度だけのmodelが
  未知taskの誤遷移を一貫してよく予測する。

`EXP-2026-008` の4次元非対角tanh RNNでは、両群のraw autonomous countを
全seedで16に一致させたまま、orthant-box認証countを16.0対10.2へ分離した。
続く `EXP-2026-009` の未使用30 seedでは、raw count一定のまま、低外乱で
robust fractionと符号記憶保持率のSpearman 0.8823、高外乱で平均marginとの
Spearman 0.9347を得た。これはP5の人工RNNにおける機構的構成例とtask接続で
ある。`EXP-2026-010` では4 network familyの未使用30 seedへ拡張し、
mean marginとのSpearman 0.8933–0.9771、raw count、coupling、局所Jacobianに
対するpooled MAE改善を確認した。`EXP-2026-011` では、各foldの対象familyを
fitから除外し、normalized marginとcertified fractionの二成分modelを新規seed
へ適用した。family別Spearmanは0.8225–0.9572で、raw countと5-feature
structural baselineに対するseed単位MAE改善も事前登録条件を満たした。

一方、candidate選択にも未使用の `modular_paired` familyを用いた
`EXP-2026-012` では、この二成分線形modelはSpearman 0、MAE 0.2238となり、
raw countとstructural baselineを上回らなかった。全120条件のraw count 16と
certificate下界違反0は維持されたため、保証そのものと、異なる構造族への
経験的較正を分離しなければならない。

さらに同familyでは、seedが変えたpair符号はtask-preservingな座標変換で
同値だった。このため30 seedの有効構造標本数は1だった。生物学的個体差を
model化する際も、parameterが異なることではなく、機能を保存する対称性で
割った後に異なる回路であることを確認する必要がある。

この二成分は、core状態の平均安全余裕と、要求budgetを満たす状態割合を分離
する。工学的には「必須状態を持つこと」と「その状態および余剰状態が利用可能
であること」を同時に設計する必要性を示す。それでも生物学的margin、発生、
遺伝、学習余剰を直接示していない。

外乱budget \(e\) に対するレパートリーを

\[
N_{\mathrm{rob}}(e)
=
\sum_k\mathbf 1[\mu_k\ge e]
\]

とし、basin probability \(p_k\) を含む安全質量を

\[
S_{\mathrm{rob}}(e)
=
\sum_k p_k\mathbf 1[\mu_k\ge e]
\]

とする。符号保持taskへの予測は4構成familyで確認し、fold内のfamily holdout
fitでも再現した。今後は、このcurveがcandidate選択にも未使用の第五family、
学習feedback、stochastic noise、cue routing、readout性能でも多変量baselineを
超えて予測するか検証する。

### 予測P6: 同一energyでの空間選択性

同じ総介入energyなら、core marginを直接消費するbridgeまたは受信座標へ
選択的に作用する場は、全再帰結合を一様に弱める場よりcore保持を高め、
reserve内部の計算・記憶を残す。

`EXP-2026-015` は一つの人工RNN familyでこの予測を支持した。次に受容体mapを
学習する局所場、低rank場、model-predictive controlを比較し、単純な
feedback-only gateに固有の結果でないことを確認する。

反証条件:

- 未知family・未知taskで、energy-matchedなglobalまたは低rank制御に対する
  core保持とreserve能力の同時優位が消える。
- 場の遅延、model誤差、飽和を加えると安全certificateが経験保持を下回る。

## 5. 実験ロードマップ

### 段階A: 人工RNNでの機構的実証

32から256状態のRNNについて、次の4群を同じparameter、標本、更新、energy
budgetで比較する。

1. taskごとに直接最適化した固定回路
2. random初期回路を生涯学習する群
3. 圧縮generatorがcore回路を生成し、全結合を学習する群
4. 圧縮generatorがcoreとplastic reserve maskを同時生成する群

主要評価:

- 学習前core性能
- 新規taskのsample efficiency
- core保持率とcatastrophic forgetting
- 自律・入力条件付きアトラクタatlas
- replica同期、最大条件付きLyapunov指数、IPC
- 固定readout retentionとtask-specific機能同値類
- 更新エネルギー、活動energy、配線量
- 変調場の空間帯域、時定数、受容体map、介入energy
- genotype記述長と生成回路の複雑度

### 段階B: 発生scheduleの介入

- dense-to-sparse pruning
- module形成
- plasticity critical period
- 細胞型別またはmodule別の更新則
- 局所・広域変調場と受容体分布の共同発達

をfactorial designで操作し、CA3型の発生所見と対応付ける。

### 段階C: 進化loopと生涯学習loop

外側loopでは \(g,P,M\) のみを進化させ、内側loopでは未知taskを学習させる。
outer loopへtest taskを漏洩させず、進化的overfittingを検出する。

### 段階D: 生物データによる外部妥当性

- 発生段階別connectome
- 細胞型別遺伝子発現
- 可塑性window
- neuromodulator、astrocyte、細胞外イオンの時空間計測
- 行動獲得時期

から、本モデルが予測する保存core、可変edge、疎化、module化の順序を検証する。

## 6. 人間規模へ進むための条件

人間規模の必要条件を単一のニューロン数やアトラクタ数から導くことはできない。
少なくとも次の軸を分離したscale lawが必要である。

1. task familyの組合せ複雑度
2. 有効状態次元と読み出し可能次元
3. 時間尺度の階層数
4. core保持下のplastic reserve
5. sensorimotor bandwidth
6. energy、遅延、配線長
7. 損傷、noise、分布shiftへの頑健性
8. 発生generatorの記述長
9. 空間変調の通信bandwidth、時定数、介入energy
10. module分割の同定誤差とcomponent certificateの合成可能性

さらに、多数のreserve moduleがcoreへ結合する場合には、

\[
\sum_{j=1}^{J}L_{f,j}\overline R_j
\le
(1-L_c)\varepsilon
\]

というworst-case保護certificateが候補になる。module数を増やすだけでは
feedback総量も増え得るため、疎結合、正規化、gatingを同時にscaleさせる
必要がある。これは人間規模の必要条件ではなく、収縮core modelにおける
十分条件である。

多重安定coreでは、必須機能 \(k\) の安全集合 \(S_k\) が外力norm
\(\mu_k\) までロバストであり、module \(j\) の活動上界を \(R_j\)、
coreへの結合を \(G_j\) とすると、

\[
\sum_{j=1}^{J}
\|G_j\|R_j
\le
\min_k\mu_k
\]

が全必須安全集合を保つworst-case十分条件になる。これは収縮coreの偏差許容
budgetと異なり、複数アトラクタ間の誤遷移を防ぐbudgetである。

人間規模では \(\min_k\mu_k\) が機能数とともに極端に小さくならないための
hierarchy、局所gating、task条件付き結合が必要になる可能性がある。ただし、
これは現時点の設計仮説であり、生物学的必要条件でも実測scale lawでもない。

空間gate \(g_t\) を含む多重安定coreでは、reserve-to-core行列を \(G\)、
reserve活動を \(r_t\)、model誤差上界を \(e_i\) として、例えば

\[
\sup_t\max_i
\left(
\left|\left[D(g_t)G r_t\right]_i\right|+e_i
\right)
\le
\min_i\mu_i
\]

を座標別の十分条件候補とする。規模を増やすには、この条件だけでなく、場を
形成・伝送するenergyと遅延、受容体mapの記述長、局所制御のmodel誤差も同時に
scaleさせなければならない。

また全機能が同じ最低marginを必要とするとは限らない。外乱budget別の
\(N_{\mathrm{rob}}(e)\) と \(S_{\mathrm{rob}}(e)\) をscaleさせることで、
少数の極端に脆弱な機能が全体のworst-case boundを支配する問題を分離する。

独立な \(m\) moduleがすべて同時に成功する必要があり、各moduleの局所成功率が
\(r\) なら、全系成功率は零次近似で

\[
q=r^m
\]

となる。従って目標 \(q\) を保つ局所必要率は

\[
r\ge q^{1/m}
\]

である。例えばmodule数が増えるほど各局所回路には1に極めて近い信頼度が
必要となる。これは人間脳の数値的必要条件ではないが、大規模系では単なる
module追加だけでなく、冗長性、誤り訂正、階層的gating、失敗相関の制御が
必要になることを示す構成的なscale下限である。

`EXP-2026-016` は、この零次積則を弱結合・非対称・異なるmodule sizeへ移す際、
局所margin、方向別負荷、size imbalanceを保持する表現が絶対保持率の較正に
役立つ一例を示した。`EXP-2026-017` は、(i) task-freeでmodule分割を回復し、
(ii) 3 moduleでcertificate合成を局所16 orthantへ因子化する構成を、強い一意
affinity gapを持つ7次元人工familyに限って確認した。`EXP-2026-018` はさらに
同じpartitionを保証するentrywise誤差半径を与えた。人間規模へ使うには、
(iii) stochastic外乱とcue/readout taskでも外挿誤差が増大しない、(iv) 半径外で
誤同定された分割に対する機能誤差上界を持つ、(v) module数増加時に局所次元、総流入load、局所
失敗率を規模非依存に制御できる、という追加条件が必要である。(i)と(ii)も一般条件
ではなく構成例であり、分割不変半径を除く(iii)–(v)は未証明である。

候補となる必要条件は、例えば

\[
d_{\mathrm{accessible}}
\ge
C_{\mathrm{required}}
\]

という容量条件だけでなく、

\[
R(B,\varepsilon)\ge R_{\mathrm{environment}},
\qquad
F_{\mathrm{core}}\ge F_{\min},
\qquad
E_{\mathrm{total}}\le E_{\max}
\]

を同時に満たす領域として表現する。

まず小規模系で各量のdimension、task数、module数に対するscaling exponentを
推定する。その後、異なるnetwork familyと生物データで外挿誤差を評価する。
人間規模への数値外挿は、この外部検証を通過するまで行わない。

## 7. 「立証」の判定基準

本構想を支持するには、次の三段階が必要である。

1. **機構的十分性**  
   発生generatorとplastic reserveを持つ人工系が、予測したtrade-offと
   優位性を再現する。
2. **一般性**  
   RNN、spiking network、異なるtask family、未知seedでも同じ法則が残る。
3. **生物学的予測力**  
   発生connectomeや行動獲得の未使用データを、単純なrandom growthや
   全結合学習modelよりよく予測する。

これらを満たしても「生物が存在する理由」を数学的に一意証明するものではない。
主張可能なのは、進化的に圧縮された機能コアと可塑的余剰が、効率、頑健性、
適応性を両立する十分な設計原理であることまでである。
