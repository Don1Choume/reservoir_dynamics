# 研究方向: 生得的機能コアと可塑的力学余剰

最終確認: 2026-07-30  
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
あるが、生物学的margin、発生、遺伝、学習余剰を直接示していない。

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

とする。符号保持taskへの予測は一つのfamilyで確認した。今後は、このcurveが
別topology、学習feedback、noise、cue routing、readout性能でもcouplingや
局所安定性baselineを超えて予測するか検証する。

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
- genotype記述長と生成回路の複雑度

### 段階B: 発生scheduleの介入

- dense-to-sparse pruning
- module形成
- plasticity critical period
- 細胞型別またはmodule別の更新則

をfactorial designで操作し、CA3型の発生所見と対応付ける。

### 段階C: 進化loopと生涯学習loop

外側loopでは \(g,P,M\) のみを進化させ、内側loopでは未知taskを学習させる。
outer loopへtest taskを漏洩させず、進化的overfittingを検出する。

### 段階D: 生物データによる外部妥当性

- 発生段階別connectome
- 細胞型別遺伝子発現
- 可塑性window
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

また全機能が同じ最低marginを必要とするとは限らない。外乱budget別の
\(N_{\mathrm{rob}}(e)\) と \(S_{\mathrm{rob}}(e)\) をscaleさせることで、
少数の極端に脆弱な機能が全体のworst-case boundを支配する問題を分離する。

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
