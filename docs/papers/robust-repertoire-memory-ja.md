# アトラクタ数を越えたリザバー評価

## ロバスト・レパートリー余裕による外乱下記憶性能の予測

研究草稿 v0.7
2026年8月2日

## 要旨

リザバー計算に用いる力学系の能力を理解するうえで、アトラクタの数は直観的な指標である。しかし、自律系で同数のアトラクタを持つ二つの系が、入力、雑音、学習済みmoduleからのfeedbackを受けたときにも同数の状態を安全に利用できるとは限らない。本研究では、離散時間tanh RNNの各符号orthantについて、成分別有界外乱下でロバスト正不変となる共通境界hyperboxを構成し、その最大一様外乱余裕を導出した。外乱budget \(e\) に対して認証可能なアトラクタ数をロバスト・レパートリー \(N_{\mathrm{rob}}(e)\) と定義し、生の自律アトラクタ数から分離した。

4次元RNNを用いた段階的な検証では、まず未使用30 seedで自律アトラクタ数を全条件16に一致させたまま、認証レパートリーを分離した。次に、全16 orthantと全16一定corner外乱方向からなる符号記憶taskを導入した。単一familyの未使用30 seedでは、低外乱で認証robust fractionと保持率のSpearman相関が0.8823、高外乱で平均marginとの相関が0.9347だった。さらにdense symmetric、sparse symmetric、asymmetric dense、feedforward non-normalの4 familyを事前登録条件で確認したところ、全480 network条件のraw countが16のまま、平均marginと保持率の相関は0.8933–0.9771となった。

最後に、各foldの対象familyをfitから除外し、他3 familyだけで標準化ridge回帰をfitするfamily・seed二重holdout確認を行った。要求外乱で無次元化した平均marginと認証robust fractionの二成分modelは、新規30 seedのpooled MAE 0.0822、family別Spearman 0.8225–0.9572を示した。seed単位paired bootstrapで、raw countおよび5-feature structural baselineのMAEよりそれぞれ0.0486 [0.0468, 0.0505]、0.0362 [0.0319, 0.0404]小さかった。

さらに候補選択にも未使用の第五の独立pair-module familyへ同じmodelを適用した。全120条件のraw countは16、30,720 challengeのcertificate違反は0だったが、二成分modelのSpearmanは0、MAEは0.2238で、raw countとstructural baselineを上回らなかった。Normalized margin単独は副次的にSpearman 0.8944を示した。またpair符号だけを変える30 seedは符号座標変換で同値であり、bootstrap区間は一点へ退化した。

そこで既存5 familyを符号座標共役で監査したところ、600 network条件は192構造classへ縮約された。各gainの30 seedに対する有効class数はdense symmetric 8、sparse symmetric 2、asymmetric dense 29、feedforward non-normal 8、modular paired 1だった。続いてmodule結合絶対値をseedごとに変えた60個の有効構造networkで積則を事前登録確認した。240条件、61,440 challengeにおいてfull固定点数は常に4×4=16、全corner符号保持率のmodule積に対する最大残差は0、component certificate下界違反は0だった。保持率は0.390625–1.0に分布した。

この独立moduleを零次基準とし、infinity normが0–0.04の対称cross-module結合を加えた。pilot後に効果量閾値を固定し、未使用30 seedの1,440点、368,640 challengeで確認した。raw固定点数は全点16のまま、平均絶対task積則残差は0から0.0831へ単調非減少し、strength 0.04の非零残差率は0.65、最大絶対残差は0.375だった。座標別component rectangleを実際のcross-edgeへ移送するcertificateと、結合normだけでmarginを減算するcertificateは、いずれもtask保持率の下界を保った。

次に、非対称bridgeを持つ2+2・2+3 node系だけで方向別component-aware ridgeをfitし、係数を固定したまま未使用30 seedの3+5 node系へ外挿した。960点、983,040 challengeでcomponent-aware MAEは0.01232、global profileは0.03970、isolated task積だけのmodelは0.01699だった。seed単位bootstrapによるbaseline minus component MAEの95%区間下限は0.02474と0.004166で、certificate chainとtask下界に違反はなかった。一方、Spearmanはcomponent-aware 0.8116に対し二baselineが0.8542、0.9062であり、順位予測の支配は得られなかった。

さらに3+5 nodeの非対称core–reserve RNNへ有界拡散型の空間変調場を導入し、reserve-to-core bridgeだけを局所gateする介入を、同じFrobenius介入energyの一様global recurrent gainと比較した。未使用30 seed、270条件、2,851,200座標遷移で局所場は8必須core orthantを全条件保持した。global対照に対するcore保持差は0.3472 [0.2393, 0.4681]、reserve線形記憶容量差は0.07683 [0.06350, 0.09123]で、certificate違反は0だった。

以上は、有限外乱下で利用可能な記憶状態を評価するにはアトラクタ数だけでなくmargin profileが必要である一方、既知構造で学習した低次元の線形較正を任意の力学系へ普遍化できないことを示す。またraw seed数をtask保存対称性で割り、component別profile、方向別結合負荷、積則残差、移送certificateを保持し、局所介入へ接続する必要がある。本結果は低次元tanh RNN、符号保持と線形記憶、既知二module分割、hyperbox十分条件、単純なglobal対照に限定される。

キーワード: reservoir computing、multistability、robust invariant set、survivability、attractor repertoire、recurrent neural network

## 1. はじめに

Reservoir Computing Generalizedを含む近年の理論は、計算基材を特定のEcho State Networkへ限定せず、多様な入力駆動力学系とreadoutの組として計算を捉える方向を示している [1,2]。この一般化は、任意の力学系を候補reservoirとして分析し、その構造とダイナミクスをtaskへ適合させる設計問題を重要にする。

多重安定系では、アトラクタ数、吸引域、遷移障壁、条件付き安定性が記憶と状態遷移の自然な記述量になる。しかし、アトラクタが自律系に存在することと、入力または外乱下でその状態を計算資源として安全に利用できることは同じではない。Basin stabilityは摂動後にどのアトラクタへ漸近するかを測り、survivabilityは過渡軌道が望ましい領域を一度も逸脱しない確率を測る [3,4]。また、非正規結合は局所固有値だけでは捉えにくい高速な過渡増幅を生じ得る [5]。したがって、生のアトラクタ数だけで有限外乱下の利用可能性を代表させることには理論的な不足がある。

本研究の問いは次の五点である。第一に、非対角tanh RNNの各符号orthantへ任意方向有界外乱に対する計算可能な安全marginを与えられるか。第二に、raw autonomous attractor countを一致させてもrobust repertoireを分離できるか。第三に、そのmargin profileが未知seedの外乱下記憶性能をraw count、coupling、局所Jacobianよりよく予測するか。第四に、局所marginと方向別結合負荷を保持するcomponent表現が、小さい非対称系から異なるmodule sizeへ外挿できるか。第五に、marginを消費する結合へ局所的に介入すると、同一energyの一様介入より必須アトラクタとreserve記憶を両立できるか。

主な貢献は以下である。

- 符号変換したtanh RNNに対し、共通境界orthant hyperboxがロバスト正不変となる一様外乱marginを導出した。
- raw count、認証count、外乱強度別robust repertoire curve、margin分布を分離する分析APIを実装した。
- count-matchedな未使用seed確認と、122,880 challengeからなる4 familyの事前登録task確認を実施した。
- 保証下界と経験的点予測を分離し、coupling、局所Jacobian、固定点座標、非正規性を比較baselineとして実装した。
- 対象familyのlabelをfitから除外するfamily・seed二重holdoutで、robust repertoire二成分modelの移送性能を事前登録確認した。
- 符号座標共役の証人探索とclass監査を実装し、既存600 network条件の実効構造多様性を192 classと算定した。
- 符号共役で異なる60個の異質独立module networkで、固定点、component certificate、全corner task保持率の積則を事前登録確認した。
- 弱いmodule間結合に対する座標別rectangle保証とnorm損失上界を導出し、raw count一定の未使用30 seedで積則残差のcoupling応答を事前登録確認した。
- 方向別component margin・結合負荷・size imbalanceを用いる固定予測器を実装し、2+2・2+3系から3+5系へのMAE外挿優位と、その順位相関上の限界を事前登録確認した。
- 有界拡散場、時変row gateのcore保護条件、energy-matched global対照を実装し、非対称3+5 core–reserve RNNで局所介入の因果的優位を事前登録確認した。

## 2. 関連研究

### 2.1 Reservoir computingの一般性とtask依存性

Reservoir computingの普遍近似理論は、fading memory filterや確率入力を含む設定で整備されてきた [2,6]。一方、fading memoryを持たないreservoirでも右無限時間operatorを用いることで近似可能性が得られる例が示されている [7]。これらは多重安定な基材をreservoir候補から排除しない理論的背景になるが、個々のアトラクタをtask上どの程度安全に利用できるかは与えない。

Topologyと性能の関係もtask依存である。2026年のrandom reservoir比較では、対称性の効果が予測対象の力学系によって変化した [8]。Photonic reservoirの最新研究でもsmall-world構造の利得が報告された一方、memoryと予測で最適parameterは一致しなかった [9]。従って、一つのtopologyまたは一つの静的指標から普遍的な設計則を結論しないことが重要である。

### 2.2 アトラクタ、有限時間機能、過渡安全性

Basin stabilityは非線形系の大域安定性を初期条件分布に基づき評価する [3]。Survivabilityは安全領域からの過渡逸脱を別に測る [4]。離散時間非線形系のinput-to-state stabilityは、有界入力に対する状態応答を扱う一般枠組みを与える [10]。

固定点の存在と有限時間機能も一致しない。2026年の連想記憶modelでは、平衡論的な記憶状態が消える容量超過後にもslow regionによるtransient retrievalが残り得ることが示された [11]。逆に、本研究が扱う問題は、固定点が存在していても有限外乱に対する安全性が不足し得るという補完的な側面である。

### 2.3 非正規性と神経回路

非正規な再帰結合は、臨界固有値によるdynamical slowingとは異なる高速な過渡増幅を作る [5]。2026年の非相反Wilson–Cowan network研究は、feedforward結合とcyclic結合が過渡reactivityとnoise駆動遷移を異なる形で組織することを報告した [12]。この知見を踏まえ、本研究は対称familyだけでなくasymmetricおよびfeedforward non-normal familyを外的妥当性検証へ含める。

### 2.4 神経系の時空間変調

神経回路の調整信号は空間的に一様とは限らない。局所的で短いGABA・glutamate入力が広く長いastrocyte Ca応答へ統合されること [18]、locus coeruleusが海馬astrocyteの秒単位の求心的統合を調整すること [19]、局所細胞外Caがsubsecondでstriatal cholinergic interneuronとdopamine放出を変えること [20] が報告されている。AstrocyteのNaおよびClも細胞・状態・細胞内位置により不均一である [21,22]。またACh–dopamineの時空間wave [23] と、dopamineがplasticityとexcitabilityを介してlatent behavioral attractorを形成・顕在化させる例 [24] がある。

これらは複数の時定数と局所性を持つ変調の存在を支持するが、本稿のcore–reserve分解、拡散場方程式、局所gateの最適性を直接示すものではない。本稿では生物機構を同定せず、空間選択的な制御が必須アトラクタのmarginとreserve能力を両立し得るかを人工系で検証する。

### 2.5 Component合成とdynamical motif

相互結合した非線形subsystemについて、ISS Lyapunov関数と方向別gain operatorから全体系の安定性を構成するsmall-gain理論がある [25]。近年は、小規模subsystemで得たneural certificateを類似構造の大規模networkへ合成する枠組みも提案されている [26]。Attractor networkではmodularityがbasin体積と収束時間を改善し得る一方、最適値は中間的であり、module化を単調な利得とはみなせない [27]。

Multi-task RNNでattractor、decision boundary、rotation等のdynamical motifがtask間で再利用されるという結果 [13] は、global scoreをcomponent atlasへ分解する計算論的動機になる。ただし、これらの研究は本稿の符号記憶taskの積則、方向別hyperbox chain、2+2・2+3から3+5への予測外挿を証明しない。本稿はこの限定されたfamilyで、保証合成と経験較正を別々に検証する。

## 3. 理論

### 3.1 対象系

次の離散時間tanh RNNを考える。

\[
\boldsymbol{x}_{t+1}=\tanh(W\boldsymbol{x}_t+\boldsymbol{\eta}_t),
\qquad \|\boldsymbol{\eta}_t\|_\infty\le e.
\]

ここでtanhは成分ごとに作用する。対象orthantを \(\boldsymbol{s}\in\{-1,1\}^d\)、符号行列を \(D_{\boldsymbol{s}}=\mathrm{diag}(\boldsymbol{s})\) とし、\(\boldsymbol{y}=D_{\boldsymbol{s}}\boldsymbol{x}\)、\(W_{\boldsymbol{s}}=D_{\boldsymbol{s}}WD_{\boldsymbol{s}}\) と変換する。

### 3.2 共通境界orthant box

変換座標で

\[
\mathcal{B}_{\boldsymbol{s}}(m)=[m,1]^d,\qquad 0<m<1
\]

を考える。行 \(i\) の正・負結合和を

\[
P_i=\sum_j\max((W_{\boldsymbol{s}})_{ij},0),\qquad
N_i=\sum_j\min((W_{\boldsymbol{s}})_{ij},0)
\]

とする。任意の \(\boldsymbol{y}\in[m,1]^d\) と \(\|\boldsymbol{\eta}\|_\infty\le e\) に対して、成分 \(i\) のtanh入力の下界は \(P_i m+N_i-e\) である。従って、

\[
P_i m+N_i-e\ge \operatorname{atanh}(m)
\]

が全成分で成立すれば、単調性により次状態も \([m,1]^d\) に含まれる。

### 3.3 最大一様外乱margin

上の十分条件からorthant \(\boldsymbol{s}\) の共通境界box marginを

\[
\mu_{\boldsymbol{s}}
=
\max_{0<m<1}
\min_i
\left[
P_i m+N_i-\operatorname{atanh}(m)
\right]
\]

と定義する。\(\mu_{\boldsymbol{s}}\ge e\) なら、\(\mathcal{B}_{\boldsymbol{s}}(m)\) は任意の時変成分別外乱 \(\|\boldsymbol{\eta}_t\|_\infty\le e\) に対してロバスト正不変である。これは十分条件であり、認証不能はアトラクタ不存在またはtask失敗を意味しない。

### 3.4 ロバスト・レパートリー

発見された自律アトラクタ集合を \(\mathcal{A}\) とする。外乱budget \(e\) に対するrobust repertoireを

\[
N_{\mathrm{rob}}(e)
=
\sum_{k\in\mathcal{A}}\mathbf{1}[\mu_k\ge e]
\]

と定義する。初期条件分布に対するbasin probabilityを \(p_k\) とすれば、

\[
S_{\mathrm{rob}}(e)
=
\sum_k p_k\mathbf{1}[\mu_k\ge e]
\]

は到達しやすさを含む認証安全質量である。本稿のtaskでは全orthantを等重みでchallengeするため、主要featureとして外乱別robust fractionと

\[
\bar{\mu}
=
\frac{1}{|\mathcal{A}|}
\sum_{k\in\mathcal{A}}\max(\mu_k,0)
\]

を用いる。

### 3.5 Curveの高さと面積

正規化robust repertoire curveを

\[
R(e)=N_{\mathrm{rob}}(e)/|A|
\]

とする。marginが有限なら、非負確率変数のtail積分公式と同じ恒等式により、

\[
\bar\mu=(1/|A|)
\sum_{k\in A}\max(\mu_k,0)
=\int_0^\infty R(z)\,dz
\]

である。要求外乱budget \(e>0\) で無次元化すると、

\[
\bar\mu/e=\int_0^\infty R(eu)\,du
\]

を得る。従って、認証robust fraction \(R(e)\) は要求点でのcurveの高さ、
normalized mean margin \(\bar\mu/e\) は要求scaleで測ったcurveの面積である。
後述のrobust pairは、局所的な閾値通過割合とレパートリー全体の余裕総量を
同時に表す。

### 3.6 弱いmodule間結合と座標別rectangle

独立moduleのblock diagonal行列を \(W_0\)、cross-module摂動を \(E\) とし、
\(W=W_0+E\) とする。符号変換後の行列を \(A=D_sWD_s\)、各座標の下側境界を
\(b_i\) とすると、rectangleの一様外乱marginは

\[
M(b;W,s)=\min_i\left\{
\sum_j\left[
\max(A_{ij},0)b_j+
\min(A_{ij},0)
\right]-\operatorname{atanh}(b_i)
\right\}
\]

である。各cross係数の最悪寄与を絶対値で抑えると、

\[
M(b;W_0+E,s)\ge M(b;W_0,s)-\|E\|_\infty
\]

を得る。従ってisolated component rectangleのmarginが
\(e+\|E\|_\infty\) 以上なら、結合後にも外乱 \(e\) に対する十分条件を保つ。
実際のcross-edge符号を使ってmarginを再計算するtransported certificateと、
normだけを使うshifted certificateを分離して評価する。

結合後の有限時間保持率を \(T_E(e)\) とし、isolated module積に対する残差を

\[
r_T(E,e)=T_E(e)-T_1(e)T_2(e)
\]

と定義する。有限challenge集合では保持率は結合強度に対して階段状になり得る
ため、微分係数ではなくsigned残差、絶対残差、非零残差率を用いる。

### 3.7 方向別component結合と合成certificate

非対称な二module系を

\[
x^+=\tanh(Ax+By+\eta_x),\qquad
y^+=\tanh(Cx+Dy+\eta_y)
\]

とする。隔離moduleのrectangle marginを \(M_x,M_y\)、受信方向別のcross loadを

\[
L_x=\|B\|_\infty,\qquad
L_y=\|C\|_\infty
\]

とすれば、tanhの単調性とcross項の三角不等式から結合後marginはそれぞれ
\(M_x-L_x\)、\(M_y-L_y\) 以上である。従って外乱budget \(e\) に対して

\[
M_x\ge e+L_x,\qquad M_y\ge e+L_y
\]

は方向別十分条件となる。実cross-edgeの符号と各座標境界を用いるtransported
rectangle、方向別norm、全cross負荷を一つのglobal normへ潰す条件の認証率を
それぞれ \(R_{\mathrm{rect}},R_{\mathrm{dir}},R_{\mathrm{global}}\) とすると、本稿の
hyperbox構成では

\[
T_{\mathrm{coupled}}\ge R_{\mathrm{rect}}
\ge R_{\mathrm{dir}}\ge R_{\mathrm{global}}
\]

を得る。最初の不等式は経験taskに対する保証照合、残りは情報を捨てる順序に
対応する。このchainは一般の非矩形安全集合に対する必要条件ではない。

高次元で全 \(2^d\) cornerを列挙する代わりに、moduleの符号に相対的な4個の
共有方向code \(k\) を用いた。零結合時にはcode別成功率 \(T_{x,k},T_{y,k}\) から

\[
T_0=\frac{1}{K}\sum_{k=1}^{K}T_{x,k}T_{y,k},\qquad K=4
\]

が厳密に成り立つ。これは全方向一様分布の完全列挙ではなく、規模外挿を可能に
する固定challenge設計である。

### 3.8 有界空間変調場と局所gate

nodeごとの変調状態を \(z_t\in[0,1]^n\)、sourceを
\(s_t\in[0,1]^n\) とし、

\[
z_{t+1}=(1-\alpha-\beta)z_t+\alpha Pz_t+\beta s_t
\]

とする。\(P\) が非負row-stochastic、\(\alpha,\beta\ge0\)、
\(\alpha+\beta\le1\) なら右辺はhypercube内の点の凸結合であり、
\(z_t\in[0,1]^n\) は正不変である。局所gateを \(g_t=1-z_t\) とし、
reserve-to-core行列 \(G\) の受信行だけを \(D(g_t)G\) に置き換える。

各core座標の双安定安全marginを \(\mu_i\)、unmodeled load上界を \(e_i\) と
すれば、

\[
\sup_t\left(
\left|[D(g_t)G r_t]_i\right|+e_i
\right)\le\mu_i
\]

は、その座標の符号安全集合を保つ十分条件である。比較するglobal制御は
\(qW\) とし、局所介入との差のFrobenius二乗energy
\(\|W-D(g_t)W\|_F^2\) と
\(\|W-qW\|_F^2\) を一致させる。これにより局所性の効果を、単なる介入量の差から
分離する。ただしこのglobal対照は一様scalarであり、最適制御ではない。

## 4. 方法

### 4.1 段階的検証

研究は探索と確認を分離した。EXP-2026-007ではcoupling 0.04と0.08を比較したが、raw countも平均2.1333低下し、count-matched判定は不成立だった。この陰性結果を保持し、事後探索でraw countを保つ0.07を選んだ。

EXP-2026-008では未使用seed 501–530を用い、coupling 0.04と0.07の両群で全seedのraw countを16に一致させた。EXP-2026-009ではdiscovery seed 401–430で外乱強度とfeatureを選び、未使用seed 601–630で符号記憶予測を確認した。EXP-2026-010ではpilot seed 801–808でfamily別条件を選び、discovery seed 801–830でpredictorをfitし、未使用seed 901–930で4 familyを確認した。

EXP-2026-011のpilotは観測済みseed 801–830をfit、901–930をtestに用い、6 candidateからpooled leave-one-family-out MAE最小のrobust pairを選択した。その後、penalty、feature、baseline、閾値を固定し、他3 familyの既観測60 seedでfitしたmodelをheld-out familyの未使用seed 1201–1230へ適用した。

EXP-2026-012では、EXP-2026-011で固定したfeature、penalty、baseline、閾値を変更せず、既知4 familyの既観測seed 801–830と901–930、計960条件でfitした。候補選択にもparameter調整にも使用していない `modular_paired` familyの未使用seed 1301–1330、計120条件を一度だけ評価した。

AUDIT-2026-001では性能taskを再実行せず、EXP-2026-011/012の重み600条件を符号対角共役 \(W'=DWD\) でclass分割した。EXP-2026-013ではtask前に各gain30 classを確認した `modular_heterogeneous` のseed 1401–1430を用い、2 global gain、4外乱の240条件を一度だけ評価した。

EXP-2026-014ではseed 1501–1510のpilotでcross strength 0–0.04のgridを評価し、結果観測後、未使用confirmation seedには触れずに平均絶対残差0.05、非零残差率0.5、集約曲線単調非減少の3経験判定を固定した。seed 1601–1630の構造gateで各internal gain・strengthの30 networkが30符号共役classになることを確認してから、同じgridのconfirmationを一度だけ評価した。

EXP-2026-015では開発専用seedでfeedback gainとnoise gridを較正した後、pilot seed 1801–1810を一度評価し、確認効果量閾値を固定した。確認taskの前にseed 1901–1930、3 feedback gainの各群が30絶対値構造classを持ち、3+5 module size、非対称性、双方向bridgeを満たすことを監査した。その後、未使用30 seed、3 feedback gain、3 noiseの270条件を一度だけ評価した。

EXP-2026-016では、pilot seed 2001–2010の2+2・2+3 moduleを用いて3予測器をfitし、係数・標準化統計・ridge penaltyをartifactへ固定した。pilot結果観測後、confirmation MAE 0.03以下、Spearman 0.75以上、global minus componentおよびproduct-only minus component MAEのbootstrap区間下限0.01、0.002以上を固定した。確認task前にseed 2101–2130の3+5 moduleについて、各internal gain・cross strength群が30絶対値構造class、非対称、双方向かつ正cross normであることを監査した。その後は再fitせず、未使用30 seedの960点を一度だけ評価した。

### 4.2 Network family

全networkは4次元、対角自己結合1.5のtanh RNNである。

| Family | 非対角構造 | Coupling gain | 外乱budget |
|---|---|---|---:|
| Dense symmetric | 全無向edge、seed別符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Sparse symmetric | 次数2の無向ring、seed別符号 | 0.04, 0.06, 0.08, 0.10 | 0.12 |
| Asymmetric dense | 全有向edge、方向別seed符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Feedforward non-normal | 上三角有向edge、seed別符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Modular paired | 独立2-node対称pair二個、pair別seed符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Modular heterogeneous | 独立2-node対称pair二個、pair別絶対値と符号 | 0.05, 0.07 | 0.08, 0.12, 0.16, 0.20 |
| Weakly coupled modular | 上記二moduleと二つの対称cross bridge | internal 0.05, 0.07; cross 0–0.04 | 0.08, 0.12, 0.16, 0.20 |

Sparse familyでは共通条件に天井効果または固定点消失が生じたため、pilotで異なるcoupling gridと外乱budgetを固定した。従ってfamily間の絶対保持率をtopologyの因果効果として比較しない。

Modular paired familyの重みは \(aI_4\) と二つの独立2次元blockからなる。module間結合は0であり、全系は二つの部分系の直積へ厳密に分解する。

Modular heterogeneousでは各pairの結合絶対値をglobal gainの0.75–1.25倍からseed別に生成した。task実行前監査で各gain30 networkが30符号共役classとなり、unorderedな二module絶対値pairにも重複がないことを確認した。

Weakly coupled modularではnode 0–2と1–3へseed別符号の対称bridgeを置いた。各行に一つのcross edgeしかないため、block diagonal基準との差のinfinity normはcross strengthに厳密に一致する。strength 0では二component、正strengthでは全4 nodeが一componentになる。

EXP-2026-015は別familyとして、自己結合1.5の3-node双安定coreと5-node収縮reserveを非対称かつ双方向bridgeで接続した。局所政策はreserve-to-core受信行だけを空間場でgateし、ungated政策は同じnetworkを無制御で用いた。Global政策は各時刻に局所政策と同じFrobenius介入energyを持つ一様recurrent gainを用いた。

EXP-2026-016の `asymmetric_modular` familyは、各module内部に符号付き非対称edgeを持ち、二方向bridge \(B,C\) を独立に生成した。Pilotは2+2と2+3、confirmationは3+5 nodeであり、internal gain 0.025、0.05、cross strength 0、0.01、0.02、0.04、外乱budget 0.08、0.12、0.16、0.20を用いた。Module分割は既知として評価器へ渡した。

### 4.3 固定点探索と符号記憶task

全16 orthantについて初期状態 \(0.9\boldsymbol{s}\) から自律系を500 step発展させ、最終残差 \(10^{-9}\) 以下かつ全時刻で符号を保つ場合に固定点を発見したとした。各固定点を4-bit符号記憶とみなした。

外乱方向 \(\boldsymbol{d}\in\{-1,1\}^4\) の全16 cornerを列挙し、\(\boldsymbol{\eta}=e\boldsymbol{d}\) を100 step一定に印加した。全時刻で元のorthant符号を保てば成功とした。一network・一外乱強度あたり256 challengeであり、EXP-2026-010とEXP-2026-011の各confirmation総数は122,880、EXP-2026-012は30,720、EXP-2026-014は368,640だった。

EXP-2026-016はdimensionに依存しない4個の符号相対方向codeを80 step印加した。各moduleの全符号orthantから500 stepで固定点を求め、結合系の同時符号保持を成功とした。Confirmationは960点、983,040 challengeである。これは全 \(2^d\) corner分布ではなく、零結合時のcode別積則を厳密に保つ整列taskである。

### 4.4 Predictorとbaseline

EXP-2026-010ではfamilyごとにdiscovery 120条件で切片付き単回帰をfitし、confirmation 120条件へ係数を固定して適用した。主要featureは平均margin \(\bar{\mu}\) である。Baselineはraw attractor count、coupling gain、off-diagonal infinity norm、全固定点にわたるworst local Jacobian infinity norm、minimum signed fixed-point coordinate、nonnormality commutator normとした。

EXP-2026-011ではfamily名をfeatureへ入れず、held-out familyのtask保持率をfitから
完全に除外した。各featureを学習fold内で標準化し、切片を罰しないridge回帰
（penalty \(10^{-3}\)）をfitした。選択modelのfeatureは
\(\bar\mu/e\) とcertified robust fractionである。Primary baselineはraw countと、
normalized coupling、normalized off-diagonal norm、local Jacobian、minimum
coordinate、nonnormalityからなる5-feature structural modelとした。
Normalized margin単独はsecondary baselineである。全予測を保持率の定義域
\([0,1]\) へclipした。

EXP-2026-012ではrobust pairを既知4 familyの全60 seedでfitし、外部familyへ適用した。学習family集合と評価family集合、学習seed集合と評価seed集合がそれぞれ交わらないことを評価器で検査した。Baselineとclip規約はEXP-2026-011から変更していない。

EXP-2026-016のglobal modelはdimension、raw attractor fraction、global certificate fraction、mean margin、全非対角infinity norm、normalized nonnormality、最大bridge normの7 featureを用いた。Component-aware modelはこれにisolated task積、component certificate積、directionalおよびtransported certificate率、方向別slackの平均・最小、load imbalance、size imbalanceの8 featureを加えた。Product-only modelはisolated task積だけを用いた。全modelは標準化ridge（penalty \(10^{-3}\)）で、pilot後の係数をconfirmationへ固定した。

### 4.5 統計と事前判定

Family別にSpearman順位相関とMAEを算出した。誤差差は同じseedの4 family・各4 gainを平均して一標本とし、30 seedを2,000回percentile bootstrapして95%区間を求めた。

EXP-2026-010の事前判定は、confirmation全480 networkでraw count 16、certificate下界違反0、4 familyすべてでmean marginとtask retentionのSpearmanが0.75超、pooled raw-count MAE minus margin MAEの95%区間下限が0超、の4項目である。Couplingとlocal Jacobianに対する誤差差はsecondary endpointとして事前に固定した。

EXP-2026-011では各confirmation seedの4 family・4 gainを平均して一標本とした
paired誤差差を用いた。事前判定は、raw count 16、certificate違反0、全familyで
robust-pair予測とのSpearman 0.75超、raw-count MAE minus robust-pair MAEの
95%区間下限0超、structural MAE minus robust-pair MAEの区間下限0超、の5項目
である。確認seedを観測する前にsource/test manifestを固定した。

EXP-2026-012も同じ5判定を用いたが、family別ではなく第五family全120点のSpearmanを用いた。各seedの4 gainを一標本とする2,000回paired bootstrapを事前登録した。Source/test manifest SHA-256は `30f1f7a11953dc6d8a5d1a7415ba8e12c311e718691475976883aa477295187d` である。

EXP-2026-013は予測modelをfitせず、構造gate、固定点数の積、component box不変性、global common-boundary certificateの保守性、全corner task保持率の積、component certificate下界の6判定をtolerance \(10^{-12}\) で固定した。Source/test manifest SHA-256は `85fb1caea1ebfb68db4e4f1ffd722534a9d6265a9db800e82d14df40739555fa` である。

EXP-2026-014は零結合積則、transported certificate下界、norm-shifted certificate下界、前者の後者に対する支配をpilot前に固定した。pilot後、最大strengthの平均絶対残差0.05以上、非零残差率0.5以上、strength別平均絶対残差の単調非減少を追加固定した。Confirmation source/test manifest SHA-256は `a64556683a20ab733581ad6fe71ddc37dfb27cb3138b6dde50faaf6a02efe538` である。

EXP-2026-015の主要判定は、場のhypercube不変性、energy一致、certificate違反0、構造class重複なし、非対称性、双方向bridge、局所政策の全core保持、最大feedbackでのungated優位差0.05以上、globalに対するcore保持差とreserve容量差のbootstrap区間下限がそれぞれ0.05と0.02以上、の10項目である。Coreは8 orthantすべてをchallengeし、reserveは同一確率入力に対する最大delay 6の線形記憶容量で評価した。Confirmation manifestは `823dc31303ae8d3c8c7b7610e8b6f8fcf8a297f5db240bd3a8c518989a3efca8` である。

EXP-2026-016は、零結合task積則、transported certificateのtask下界、transported–directional–global chain、feature有限性、component MAE 0.03以下、Spearman 0.75以上、global minus componentおよびproduct-only minus component MAEのseed bootstrap区間下限0.01、0.002以上、という8実質判定と、それらのaggregateを固定した。構造gate通過と固定model hash一致もtask実行前提とし、順位相関のbaseline優位は登録していない。Confirmation manifestは `f013d7f40c2e2dd146fce6d61ea5d95288b3b8b0892b4413ea8c381817ebc170`、固定model SHA-256は `db0b50a648fb085ca687922a531fab5482af2a134bd01eefc3efe3dd85675a01` である。

## 5. 結果

### 5.1 Raw countとrobust repertoireの分離

EXP-2026-008では、未使用30 seedの両群でraw autonomous countが16だった。一方、認証robust countはcoupling 0.04で16.0、coupling 0.07で10.2となり、対応差は5.8、95%区間は[5.2667, 6.2667]だった。全認証boxのsafe外乱保持率と1.1倍boundary witness escape率は1だった。

### 5.2 単一familyでのtask接続

EXP-2026-009では全confirmation条件のraw countが16だった。外乱0.08でcertified robust fractionとtask retentionのSpearmanは0.8823、未知seed MAEは0.0456だった。Raw-count baseline MAEは0.1010で、raw minus robustの誤差差は0.0554 [0.0474, 0.0647]だった。

外乱0.16ではmean marginとtask retentionのSpearmanは0.9347、未知seed MAEは0.0368だった。Raw-count baseline MAEは0.1214で、誤差差は0.0846 [0.0742, 0.0955]だった。全61,440 challengeでcertificate下界違反はなかった。

### 5.3 4 familyでの外的妥当性

EXP-2026-010の事前登録4判定はすべて成立した。全480 network条件でraw countは16、全122,880 challengeでcertificate下界違反は0だった。

| Family | Mean-margin Spearman | Margin MAE | Coupling MAE | Raw-count MAE |
|---|---:|---:|---:|---:|
| Dense symmetric | 0.8933 | 0.0424 | 0.0728 | 0.1311 |
| Sparse symmetric | 0.9244 | 0.0788 | 0.0790 | 0.1325 |
| Asymmetric dense | 0.9557 | 0.0399 | 0.0398 | 0.1522 |
| Feedforward non-normal | 0.9771 | 0.0285 | 0.0299 | 0.1143 |

Family別相関は0.8933–0.9771であり、事前閾値0.75をすべて上回った。Seed単位pooled誤差差を次に示す。

| 比較 | MAE差 | 95%区間 | 登録上の位置付け |
|---|---:|---:|---|
| Raw count minus margin | 0.0851 | [0.0828, 0.0873] | Primary |
| Coupling minus margin | 0.0080 | [0.0030, 0.0133] | Secondary |
| Local Jacobian minus margin | 0.0051 | [0.0020, 0.0085] | Secondary |

Poolした平均ではmargin predictorが三baselineを上回った。一方、feedforward non-normal family内ではlocal Jacobian MAE 0.0223とminimum coordinate MAE 0.0228がmargin MAE 0.0285より小さかった。従ってmarginが全familyで全baselineを支配したとはいえない。

### 5.4 Family・seed二重holdout

EXP-2026-011の事前登録5判定はすべて成立した。全480 confirmation条件でraw
countは16、全122,880 challengeでcertificate下界違反は0だった。Robust pairの
pooled MAEは0.0822、pooled Spearmanは0.7975だった。

| Held-out family | Robust-pair Spearman | Robust-pair MAE |
|---|---:|---:|
| Dense symmetric | 0.9572 | 0.0537 |
| Sparse symmetric | 0.8225 | 0.1289 |
| Asymmetric dense | 0.9534 | 0.0777 |
| Feedforward non-normal | 0.9469 | 0.0684 |

Pilotで最大誤差だったsparse symmetricを除外せず、全familyで事前閾値0.75を
上回った。Pooled baselineとseed単位paired誤差差を次に示す。

| Baseline | Baseline MAE | Baseline minus robust-pair MAE | 95%区間 |
|---|---:|---:|---:|
| Raw count | 0.1308 | 0.0486 | [0.0468, 0.0505] |
| Normalized margin | 0.0886 | 0.0064 | [0.0059, 0.0070] |
| Structural | 0.1184 | 0.0362 | [0.0319, 0.0404] |

Normalized margin比較はsecondary endpointである。全区間下限が0を上回ったが、
事後に主要判定へ追加していない。

### 5.5 Candidate選択から隔離したmodular family

EXP-2026-012では、事前登録5判定のうちraw count一致とcertificate下界の2件が
成立し、順位相関と二つのbaseline優位は不成立だった。全120条件でraw countは
16、全30,720 challengeでcertificate下界違反は0だった。

| Predictor | MAE | Spearman | Baseline minus robust-pair MAE |
|---|---:|---:|---:|
| Robust pair | 0.2238 | 0.0000 | — |
| Raw count | 0.2188 | 0.0000 | -0.0050 |
| Normalized margin | 0.1955 | 0.8944 | -0.0282 |
| Structural | 0.2160 | 0.5774 | -0.0078 |

Robust pairはraw countとstructural baselineの両方より大きいMAEとなった。
Normalized margin単独は高い順位相関を示したが、同じconfirmation上のsecondary
結果であり、新たに選択済みの普遍predictorとは扱わない。Task retentionの範囲は
0.5625–1.0だった。

三つのpaired bootstrap区間は、いずれも表の推定値と同じ一点へ退化した。
原因は、seedが変える二つのpair coupling符号が符号座標変換で互いに共役であり、
全orthant・全corner列挙の集約量がseed不変だったことである。従って30 seedを
独立な構造標本とは扱わない。

### 5.6 符号共役で割った有効構造多様性

AUDIT-2026-001では各family・gainの30 seedを符号対角共役で分割した。

| Family | Raw / gain | Effective / gain | Raw / effective |
|---|---:|---:|---:|
| Dense symmetric | 30 | 8 | 3.75 |
| Sparse symmetric | 30 | 2 | 15.00 |
| Asymmetric dense | 30 | 29 | 1.03 |
| Feedforward non-normal | 30 | 8 | 3.75 |
| Modular paired | 30 | 1 | 30.00 |

全20 family-gain群の600 network条件は192 class、effective fraction 0.32だった。Dense、sparse、feedforward、modularのclass数はsigned graphのcycle-rankから得る理論値8、2、8、1と一致した。Asymmetric denseは理論上512 classを許し、30 seed中29 classを観測した。

### 5.7 異質独立moduleの積則確認

EXP-2026-013では各gain30 class、計60個の異質networkを確認した。全240条件でfull raw countは16、各moduleは4だった。61,440 challengeにおける主要結果を示す。

| 判定量 | 結果 |
|---|---:|
| Full count minus module count product | 0 |
| Task retention minus module retention product | 最大絶対値0 |
| Component box minimum slack | -1.11e-16 |
| Full task minus component certified fraction | 最小0 |
| Full common fraction minus component fraction | 範囲0–0 |

task retentionは0.390625–1.0に分布した。従ってtask積則は保持率が定数だったためではない。既存common-boundary fractionは今回の4閾値でcomponent積fractionと一致したが、異質module一般での等号は主張しない。

### 5.8 弱結合による積則残差

EXP-2026-014の未使用30 seedでは、全1,440点でcoupled raw countが16だった。

| Cross strength | 平均絶対残差 | 非零残差率 | 平均signed残差 | Transported率 | Norm-shifted率 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0.6594 | 0.6594 |
| 0.0025 | 0.0123 | 0.1500 | -0.0049 | 0.6443 | 0.6385 |
| 0.005 | 0.0199 | 0.2208 | -0.0063 | 0.6333 | 0.6208 |
| 0.01 | 0.0229 | 0.2292 | -0.0097 | 0.6203 | 0.5979 |
| 0.02 | 0.0379 | 0.3792 | -0.0284 | 0.6052 | 0.5656 |
| 0.04 | 0.0831 | 0.6500 | -0.0358 | 0.5250 | 0.4406 |

平均絶対残差はstrength順に単調非減少し、最大絶対残差は0.375だった。事前登録8判定はすべて成立した。Transported rectangleとnorm-shifted certificateのtask下界違反はなく、transportedがnorm-shiftedを下回る点もなかった。従って、raw count一定でもmodule間相互作用はisolated積からの機能残差として観測でき、実際のcross-edge符号を用いることでnormだけの最悪時保証より多くのorthantを認証できた。

### 5.9 空間局所gateによるcore保護とreserve記憶

EXP-2026-015の事前固定10判定はすべて成立した。確認前構造gateでは90 networkが90絶対値構造classとなり、全て3+5 module、非対称、双方向bridgeだった。270条件、2,851,200座標遷移で局所政策の8必須orthant保持率は全て1、certificate違反は0、場の値域は[0, 0.99843]、energy-matching最大誤差は4.44×10⁻¹⁶だった。

| 比較 | 対応平均差 | 95% bootstrap区間 |
|---|---:|---:|
| Local minus global core retention | 0.347222 | [0.239317, 0.468056] |
| Local minus global reserve capacity | 0.076829 | [0.063495, 0.091225] |
| Local minus ungated core retention | 0.124074 | [0.044896, 0.213912] |

最大feedback gain 0.80では、noise 0、0.04、0.08に対するcore保持率が局所政策で全て1、ungatedで0.7875、0.7833、0.7750、globalで0.4708、0.4250、0.2792だった。従って、同じ介入energyでも一様な全結合減衰よりbridge局所gateの方が、必須core状態を保ちながらreserve内部の線形記憶を残した。これは単一のfeedback-driven局所政策と単純global対照の比較であり、局所制御一般の最適性を示さない。

### 5.10 非対称・異サイズmoduleへのcomponent外挿

EXP-2026-016の確認前構造gateでは、2 internal gain、4 cross strengthの各群で30 networkが30絶対値構造classとなり、全て3+5 module、非対称、双方向bridgeだった。固定pilot modelのhashは一致し、960点、983,040 challengeにおける事前登録9判定はすべて成立した。Task保持率は0.300781–1、零結合の最大絶対積則残差は0、全条件を含む最大絶対task積残差は0.185547だった。

| Model | Confirmation MAE | Spearman |
|---|---:|---:|
| Component-aware | 0.012318 | 0.811620 |
| Global profile | 0.039698 | 0.854236 |
| Product-only | 0.016989 | 0.906200 |

seed単位bootstrapでglobal minus component MAEは0.027380、95%区間 [0.024736, 0.030278]、product-only minus componentは0.004671、[0.004166, 0.005115]だった。従ってcomponent表現は絶対保持率の較正を改善したが、順位相関では二baselineを上回らなかった。

| Cross strength | Task保持率 | Transported率 | Directional率 | Global率 |
|---:|---:|---:|---:|---:|
| 0 | 0.85309 | 0.73223 | 0.73223 | 0.73223 |
| 0.01 | 0.85273 | 0.71914 | 0.68216 | 0.66888 |
| 0.02 | 0.84934 | 0.70749 | 0.64362 | 0.63874 |
| 0.04 | 0.83455 | 0.66943 | 0.50885 | 0.48223 |

全strengthでtask、transported、directional、globalのchain違反は0だった。実際のcross-edge情報を残すほど認証率が高く、最大strengthではtransportedがglobalより0.18720多くのchallengeを認証した。これは保証の保守性緩和であり、未認証challengeの失敗を意味しない。

## 6. 考察

### 6.1 アトラクタ数から利用可能性へ

本結果の中心は、全networkが同じ16個の自律符号固定点を持つ条件でも、外乱下保持率が大きく異なることである。Raw countは存在を数えるが、有限外乱に対する安全性を含まない。\(N_{\mathrm{rob}}(e)\) とmargin分布は、要求外乱budgetに応じて利用可能な状態を数え直す。

低外乱ではthreshold付きrobust fractionが有効であり、高外乱では連続的な平均marginがより情報を持った。EXP-2026-011では、curveの要求点での高さ \(R(e)\) と無次元化面積 \(\bar\mu/e\) の組が、いずれか単独より小さい未知seed MAEを示した。従って、分析ツールは単一scoreへ縮約せず、raw count、certified count、\(N_{\mathrm{rob}}(e)\) curve、margin分布、未認証率を保持すべきである。

### 6.2 保証と予測の分離

Hyperbox certificateは任意の時変成分別外乱に対する十分条件である。今回の一定corner taskでは、認証robust fractionは全条件で経験保持率の下界になった。ただし未認証orthantにも成功例があり、点予測としては保守的である。

従ってツールは二つの出力を分ける必要がある。第一は反例が出れば理論または実装が誤っている保証下界、第二はdataset上の相関や交差検証誤差として評価する経験予測である。この分離により、「認証不能」を「失敗」と誤解することを避けられる。

### 6.3 局所安定性と有限外乱安全性

Local Jacobianは固定点近傍の微小摂動増幅を測るが、符号境界までの有限距離とworst-case方向を直接含まない。Orthant marginは両者をhyperbox十分条件の形で含む。EXP-2026-010のpooled secondary解析でmarginがlocal Jacobianを上回ったことは、この差がtask予測に寄与する可能性を示す。

ただしfamily内では局所baselineが優位な例もあった。EXP-2026-011ではcoupling、局所Jacobian、minimum coordinate、nonnormalityを同時に含むstructural modelを事前登録baselineとし、robust pairがpooled MAEを上回った。これは線形ridgeの範囲での増分予測力であり、非線形model、family indicatorを用いた階層model、nested cross-validationとの比較は残る。

### 6.4 Module積則と較正の非普遍性

Block diagonalな二部分系では、固定点集合は直積となり、固定点数は積になる。
各部分系のrobust fractionを \(R_1(e),R_2(e)\) とすれば、全系の同時
certificate率は

\[
R(e)=R_1(e)R_2(e)
\]

である。全challengeを直積として一様列挙し、全module成功を全系成功と定義
すれば、有限時間保持率も \(T(e)=T_1(e)T_2(e)\) となる。一方、全系の
平均marginは

\[
\bar\mu=\int_0^\infty R_1(z)R_2(z)\,dz
\]

であり、部分系平均の単純な和や積ではない。

この積則はmoduleごとに異なるbox境界を許すcomponent certificateに対する。既存のglobal certificateは全座標に一つの境界を課すため、一般にはcomponent margin以下の保守的下界となる。EXP-2026-013の単純な異質2-node moduleでは評価した4閾値でglobal fractionとcomponent積fractionが一致したが、一般の等号へ強化しない。

EXP-2026-012の陰性結果はrobust repertoireの定義またはcertificate下界を否定
しない。否定されたのは、非直積な既知4 familyでfitした
\((\bar\mu/e,R(e))\) の線形較正を、積構造へそのまま移送できるという仮説で
ある。任意の力学系を扱う分析ツールには、global summaryだけでなく、
connected component、近似module、component別curve、積則残差を保持する必要が
ある。

また、pair符号のseed差がtask-preserving共役で消えたことは、parameter seed数
と有効な構造多様性が異なることを示す。外的妥当性の標本数は、taskを保存する
対称性で割ったnetwork同値類に基づいて監査すべきである。

AUDIT-2026-001では600 network条件が192符号共役classへ縮約された。一方、結合絶対値を変えたEXP-2026-013は各gain30 classを保ち、task retentionが0.390625–1.0へ変化する中で \(T(e)=T_1(e)T_2(e)\) を最大残差0で確認した。従ってcomponent積則は、EXP-2026-012の退化した符号seedだけに依存する現象ではない。

EXP-2026-014ではこの零次模型へ弱いcross couplingを加え、raw count 16を保ったまま平均絶対積則残差が0から0.0831へ増えることを未使用seedで確認した。平均signed残差は負だったが、平均絶対残差より絶対値が小さく、結合が全challengeで一方向に有害だったわけではない。従ってAtlasは結合normだけでなく、component別成功、signed/absolute積則残差、transported certificateを保持する必要がある。

EXP-2026-016は非対称bridgeとmodule size差へこの表現を外挿し、小さい二familyだけでfitしたcomponent-aware modelが未知3+5 familyでglobal profileとisolated task積より低いMAEを示した。これはcomponent分解が絶対保持率の較正へ増分情報を与える支持である。一方、順位相関は二baselineより低く、global featureの粗い順序づけを置き換えたわけではない。較正とrankingは異なる評価軸であり、「全指標で支配」と解釈しない。次は別generator、3 module以上、分割未知、stochastic/cue/readout task、非線形baselineで検証する必要がある。

### 6.5 Profileから空間局所介入へ

EXP-2026-015は、安全marginを観測指標として用いるだけでなく、marginを消費するreserve-to-core bridgeへ選択的に介入した。同一Frobenius energyのglobal制御はcore内部とreserve内部の再帰結合も一様に弱めるため、core誤遷移を十分抑えられない条件でreserve記憶も失った。局所gateはcoreへ流入するloadだけを削減し、reserve内部の時間応答を残した。

これは「局所性が常に優れる」という結論ではない。Global対照は一様scalarであり、network modelを使う低rank制御、MPC、最適なnode選択は局所政策を上回り得る。また局所政策も既知bridgeへ直接アクセスしている。次に必要なのは、同じ観測情報、model誤差、制御帯域、energyの下で、学習された受容体mapと強い制御baselineを比較することである。

### 6.6 生得的機能コアと可塑的余剰への含意

本研究の長期仮説は、生物の神経系には発生的に形成された必須機能coreと、個体学習に利用できるplastic reserveがあり、新規学習からcoreへ流入する負荷が安全marginを消費するというものである。大規模画像研究は構造connectomeへの多遺伝子的影響と、多尺度構造固有モードによる機能ダイナミクス制約を支持する [16,17] が、アトラクタ余剰を直接測ってはいない。必須機能 \(k\) の外乱またはfeedback loadを \(e_k\)、marginを \(\mu_k\) とすれば、

\[
\rho_k=\frac{e_k}{\mu_k}
\]

を安全負荷率候補とできる。\(\rho_k<1\) は本hyperbox familyにおける十分条件であり、\(\rho_k\ge1\) は必ず失敗する必要条件ではない。

EXP-2026-010は、同じ状態数を持つ人工回路でもmargin分布により利用可能性が異なることを示した。EXP-2026-011はさらに、要求budgetで残る状態割合と余裕総量の二成分が、fitから除外した構造の新規seedへ移送できることを示した。しかしEXP-2026-012では積構造への線形較正移送が失敗したため、回路のmodule構成を無視した普遍scoreとしては使えない。EXP-2026-016は局所margin、方向別load、size imbalanceを明示すれば一つの異サイズfamilyへ較正を移せることを示した。EXP-2026-015は、非対称core–reserve構造で局所的にfeedback loadを制御すれば、同一energyの一様制御よりcore保持とreserve記憶を両立できる人工的十分例を加えた。

Astrocyte、細胞外イオン、dopamine、AChに関する一次研究 [18–24] は、局所から広域、subsecondから分単位の変調が共存し得ることを支持する。しかし、いずれも本稿のcore–reserve分解または局所gate方程式を同定していない。本結果は長期仮説と整合する構成例であって、遺伝的設計、発生、可塑性、生物回路を実証していない。発生過程、受容体map、学習maskを操作し、core margin、忘却、新規task獲得の因果関係を測る別研究が必要である。

### 6.7 人間規模の条件について

現段階で人間規模の処理能力に必要なnode数、アトラクタ数、reserve次元を導くことはできない。導出可能なのは、候補となる必要条件の形式である。

第一に、環境外乱budget \(e\) に対してtask-relevantな機能同値類を覆うだけの \(N_{\mathrm{rob}}(e)\) または \(S_{\mathrm{rob}}(e)\) が必要である。第二に、必須機能の安全負荷率 \(\rho_k\) を許容範囲に保つgating、抑制、疎結合、module分離が必要になり得る。第三に、未知taskを追加する能力には、既存機能を壊さず新しい機能同値類を形成できるplastic reserveと更新energyが必要である。

独立同型な \(m\) moduleがすべて同時に成功する零次模型では、局所成功率
\(r\) と全系成功率 \(q\) の間に \(q=r^m\) が成り立つ。従って目標 \(q\)
には \(r\ge q^{1/m}\) が必要である。大規模系では局所信頼度を1へ近づける
だけでなく、冗長性、誤り訂正、階層的gating、失敗相関の制御が必要になる。
これは人間脳の実測下界ではなく、独立module仮定下の構成的scale条件である。

EXP-2026-016は2 module・最大8 nodeまでの範囲で、局所marginと方向別loadを残す表現がmodule size変更後の絶対保持率を較正できる一例を与えた。これを規模条件へ昇格するには、module分割の同定誤差、3 module以上でのcertificate合成計算量、失敗相関、stochastic外乱、cueとreadoutの影響に対する外挿誤差上界が必要である。従って人間規模のnode数またはアトラクタ数はまだ導かない。

これらは定量的な人間下界ではない。人間規模へ接続するには、task組合せ複雑度、時間尺度階層、通信bandwidth、energy、発生記述長、plastic reserve枯渇を同時に変化させるscale lawが必要である。

## 7. 限界

第一に、robust repertoireの主実験は4次元tanh RNN、component外挿は4–8次元、空間場実験は8次元tanh RNNであり、高次元、学習済み、spiking、物理reservoirへの一般化は未確認である。第二に、repertoire taskの外乱は100 step一定のcorner方向、component外挿も80 step一定の4方向codeである。EXP-2026-015は確率外乱を用いたが、別のcore制御taskであり、stochastic repertoire predictorを確認していない。Certificate自体は任意の時変成分別外乱を扱うが、経験照合の範囲は狭い。

第三に、共通境界hyperboxは保守的であり、座標別box、zonotope、polytope、level set、viability kernelより小さい安全集合しか認証しない可能性がある。第四に、raw countは16個のorthant初期値から発見した固定点数であり、全アトラクタの完全列挙ではない。第五に、sparse familyの外乱budgetとcoupling gridは他familyと異なり、topology間の絶対性能差を因果的に比較できない。

第六に、EXP-2026-010のcouplingとlocal Jacobianに対するpooled優位は事前登録secondary endpointである。第七に、EXP-2026-011はfold内で対象familyをfitから除外したが、candidate選択には4 familyすべてのpilot成績を用いた。第八に、EXP-2026-012の第五familyではrobust-pair線形modelが外挿に失敗した。さらに30 seedは符号共役で同値なため、modular family母集団に対する区間推定ではない。第九に、normalized margin単独の第五family相関は同じconfirmation上のsecondary結果であり、独立確認されていない。第十に、AUDIT-2026-001は符号対角共役だけを扱い、node permutation、一般similarity、近似共役は未監査である。第十一に、EXP-2026-013は独立な2+2 node moduleに限定される。第十二に、EXP-2026-014は対称二bridgeの弱結合だけを扱う。第十三に、EXP-2026-016は非対称・異サイズを扱ったが、既知二module分割、一generator、整列4方向task、線形ridgeに限定され、順位相関ではbaselineを上回らなかった。第十四に、EXP-2026-015のglobal対照は一様scalarで、局所政策は既知bridgeへ直接アクセスした。低rank、MPC、学習済み制御との比較、場のmodel誤差と遅延、plasticity ruleは未検証である。第十五に、別task family、basin-weighted safe mass、生物学的core–reserve仮説、人間規模条件は本稿の実験から直接導かれない。

## 8. 結論

自律アトラクタ数を完全に一致させても、有限外乱下で利用できる記憶状態は異なり得る。符号orthantごとのrobust invariant hyperbox marginを用いることで、この差を保証下界と経験予測の両面から記述できた。未使用seedと4 network familyで平均marginは外乱下符号記憶保持率と高い順位相関を持った。さらにcurveの高さと面積からなるrobust pairは、familyをfitから除外した未知seed予測でraw countと多変量structural baselineより小さいMAEを示した。

しかし、candidate選択にも未使用の独立module familyでは同じ線形modelが外挿に失敗した。さらに既存600 network条件は符号共役で192 classへ縮約された。異質独立moduleでは固定点、component certificate、task保持率の積則を確認し、対称弱結合ではraw countを保ったまま積則残差がcoupling normとともに増えることと、二種類のrectangle保証がtask下界を保つことを確認した。方向別component modelは2+2・2+3から3+5系へ再fitなしで外挿し、global profileとisolated task積よりMAEを改善したが、順位相関では上回らなかった。非対称3+5 core–reserve系では、同一介入energyの局所場が一様global制御より必須core状態とreserve記憶を保った。

従ってAtlasはraw seed数とglobal scoreだけでなく、task保存対称性で割った有効構造class、component別profile、方向別load、積則残差、移送certificate、介入場とenergyを保持すべきである。本研究の結論は「アトラクタ数から外乱budget付き・構造分解付き・介入可能なprofileへ評価単位を移す必要がある」までであり、「一つのglobal score、線形較正、局所制御が任意の力学系へ通用する」ではない。次の焦点は別generator・3 module以上・未知分割でのcomponent外挿、stochastic repertoire、高次元系、より豊かなset表現、強いenergy-matched制御baselineである。

## データ・コードと再現性

実験spec、seed、判定、導出済みartifact、主張台帳、実装、テストは本repositoryに保存した。主要記録はEXP-2026-008からEXP-2026-016とAUDIT-2026-001である。EXP-2026-011 confirmationはsource/test manifest SHA-256 `b022077a3279917d02805a021048382f0b50f33387283d5c900a82b3ff9d0fcd`、EXP-2026-012は `30f1f7a11953dc6d8a5d1a7415ba8e12c311e718691475976883aa477295187d` で事前固定した。EXP-2026-012実行前には全145テストが通過し、branch coverageは88%だった。初回出力転送失敗後、結果未観測のまま同一code・同一seedを決定論的にreplayしてartifactを回収した。EXP-2026-013は全165 testと11 subtest、branch coverage 88%を確認し、manifest `85fb1caea1ebfb68db4e4f1ffd722534a9d6265a9db800e82d14df40739555fa` を固定して一度だけ実行した。EXP-2026-014はpilot前manifest `c8d4ee905672f9ea0a8b7d841ac9deaef58b2e1be0d13c99d10959a6e066f51e` でpilotを一度実行し、3経験判定を追加固定した。Confirmation前に全171 testと11 subtest、branch coverage 88%を確認し、manifest `a64556683a20ab733581ad6fe71ddc37dfb27cb3138b6dde50faaf6a02efe538` で一度だけ実行した。EXP-2026-015はpilot後に確認閾値を固定し、確認前に全180 testと11 subtest、branch coverage 88%を確認した。Manifest `823dc31303ae8d3c8c7b7610e8b6f8fcf8a297f5db240bd3a8c518989a3efca8` で未使用30 seedを一度だけ実行した。EXP-2026-016は確認前に全186 unittest、7 DOCX test、branch coverage 87%を確認し、manifest `f013d7f40c2e2dd146fce6d61ea5d95288b3b8b0892b4413ea8c381817ebc170` と固定model hash `db0b50a648fb085ca687922a531fab5482af2a134bd01eefc3efe3dd85675a01` で未使用30 seedを一度だけ実行した。

<!-- pagebreak -->

## 参考文献

[1] Kubota A, et al. Reservoir Computing Generalized. arXiv:2412.12104, 2024. https://arxiv.org/abs/2412.12104

[2] Grigoryeva L, Ortega JP. Echo State Networks are Universal. Neural Networks 108, 495–508, 2018. https://doi.org/10.1016/j.neunet.2018.08.025

[3] Menck PJ, et al. How Basin Stability Complements the Linear-Stability Paradigm. Nature Physics 9, 89–92, 2013. https://doi.org/10.1038/nphys2516

[4] Hellmann F, et al. Survivability of Deterministic Dynamical Systems. Scientific Reports 6, 29654, 2016. https://doi.org/10.1038/srep29654

[5] Hennequin G, Vogels TP, Gerstner W. Non-normal Amplification in Random Balanced Neuronal Networks. Physical Review E 86, 011909, 2012. https://doi.org/10.1103/PhysRevE.86.011909

[6] Grigoryeva L, Ortega JP. Universal Discrete-Time Reservoir Computers with Stochastic Inputs and Linear Readouts Using Non-Homogeneous State-Affine Systems. JMLR 19, 1–40, 2018. https://jmlr.org/papers/v19/18-020.html

[7] Sugiura T, et al. Nonessentiality of Reservoir’s Fading Memory for Universality of Reservoir Computing. IEEE TNNLS 35, 16801–16815, 2024. https://doi.org/10.1109/TNNLS.2023.3298013

[8] Dhadphale A, et al. Prediction Performance of Random Reservoirs with Different Topology for Nonlinear Dynamical Systems with Different Number of Degrees of Freedom. Chaos, 2026. https://doi.org/10.1063/5.0314081

[9] Park S, et al. Photonic Reservoir Computing with Complex Networks. arXiv:2607.23285, 2026. https://arxiv.org/abs/2607.23285

[10] Jiang ZP, Wang Y. Input-to-State Stability for Discrete-Time Nonlinear Systems. Automatica 37, 857–869, 2001. https://doi.org/10.1016/S0005-1098(01)00028-0

[11] Clark DG. Transient Dynamics of Associative Memory Models. Physical Review E 113, 054301, 2026. https://doi.org/10.1103/42y2-bsh1

[12] Poggialini A, et al. Non-normal Dynamics on Nonreciprocal Networks: Reactivity and Effective Dimensionality in Neural Circuits. Physical Review E, accepted 16 July 2026. https://doi.org/10.1103/jv6l-3s5z

[13] Driscoll L, Shenoy KV, Sussillo D. Flexible Multitask Computation in Recurrent Networks Utilizes Shared Dynamical Motifs. Nature Neuroscience 27, 1349–1363, 2024. https://doi.org/10.1038/s41593-024-01668-6

[14] Dohare S, et al. Loss of Plasticity in Deep Continual Learning. Nature 632, 768–774, 2024. https://doi.org/10.1038/s41586-024-07711-7

[15] McClelland JL, McNaughton BL, O’Reilly RC. Why There Are Complementary Learning Systems in the Hippocampus and Neocortex. Psychological Review 102, 419–457, 1995. https://doi.org/10.1037/0033-295X.102.3.419

[16] Wainberg M, et al. Genetic Architecture of the Structural Connectome. Nature Communications 15, 1962, 2024. https://doi.org/10.1038/s41467-024-46023-2

[17] Xia J, et al. Multiscale Structural Connectome Eigenmodes Constrain Human Brain Functional Dynamics. Communications Biology, 2026. https://doi.org/10.1038/s42003-026-10558-5

[18] Cahill MK, et al. Local and Transient Inputs to Astrocytes Generate Larger and Longer-Lasting Calcium Signals. Nature, 2024. https://doi.org/10.1038/s41586-024-07311-5

[19] Centripetal Integration of Past Events in Hippocampal Astrocytes Regulated by the Locus Coeruleus. Nature Neuroscience, 2024. https://doi.org/10.1038/s41593-024-01612-8

[20] Rapid Astrocyte Modulation of Local Extracellular Calcium Drives Striatal Cholinergic Interneurons and Dopamine Release. Nature Communications, 2024. https://doi.org/10.1038/s41467-024-54253-7

[21] Astrocytic Sodium Homeostasis Exhibits Cellular and Subcellular Heterogeneity and Controls Potassium Uptake. Nature Communications, 2026. https://doi.org/10.1038/s41467-026-73435-z

[22] Brain-State-Dependent Astrocytic Chloride. Nature Communications, 2023. https://doi.org/10.1038/s41467-023-37433-9

[23] Acetylcholine–Dopamine Waves as a Reaction–Diffusion System. Nature Communications, 2023. https://doi.org/10.1038/s41467-023-42311-5

[24] Dopamine Builds and Reveals Latent Behavioral Attractors. Nature Communications, 2024. https://doi.org/10.1038/s41467-024-53976-x

[25] Dashkovskiy SN, Rüffer BS, Wirth FR. Small Gain Theorems for Large Scale Systems and Construction of ISS Lyapunov Functions. SIAM Journal on Control and Optimization 48, 4089–4118, 2010. https://doi.org/10.1137/090746483

[26] Zhang S, Xiu Y, Qu G, Fan C. Compositional Neural Certificates for Networked Dynamical Systems. Proceedings of L4DC, PMLR 211, 272–285, 2023. https://proceedings.mlr.press/v211/zhang23a.html

[27] Pradhan N, Dasgupta S, Sinha S. Modular Organization Enhances the Robustness of Attractor Network Dynamics. Europhysics Letters 94, 38004, 2011. https://doi.org/10.1209/0295-5075/94/38004
