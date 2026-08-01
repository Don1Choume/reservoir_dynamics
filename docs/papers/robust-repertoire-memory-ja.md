# アトラクタ数を越えたリザバー評価

## ロバスト・レパートリー余裕による外乱下記憶性能の予測

研究草稿 v0.4
2026年8月1日

## 要旨

リザバー計算に用いる力学系の能力を理解するうえで、アトラクタの数は直観的な指標である。しかし、自律系で同数のアトラクタを持つ二つの系が、入力、雑音、学習済みmoduleからのfeedbackを受けたときにも同数の状態を安全に利用できるとは限らない。本研究では、離散時間tanh RNNの各符号orthantについて、成分別有界外乱下でロバスト正不変となる共通境界hyperboxを構成し、その最大一様外乱余裕を導出した。外乱budget \(e\) に対して認証可能なアトラクタ数をロバスト・レパートリー \(N_{\mathrm{rob}}(e)\) と定義し、生の自律アトラクタ数から分離した。

4次元RNNを用いた段階的な検証では、まず未使用30 seedで自律アトラクタ数を全条件16に一致させたまま、認証レパートリーを分離した。次に、全16 orthantと全16一定corner外乱方向からなる符号記憶taskを導入した。単一familyの未使用30 seedでは、低外乱で認証robust fractionと保持率のSpearman相関が0.8823、高外乱で平均marginとの相関が0.9347だった。さらにdense symmetric、sparse symmetric、asymmetric dense、feedforward non-normalの4 familyを事前登録条件で確認したところ、全480 network条件のraw countが16のまま、平均marginと保持率の相関は0.8933–0.9771となった。

最後に、各foldの対象familyをfitから除外し、他3 familyだけで標準化ridge回帰をfitするfamily・seed二重holdout確認を行った。要求外乱で無次元化した平均marginと認証robust fractionの二成分modelは、新規30 seedのpooled MAE 0.0822、family別Spearman 0.8225–0.9572を示した。seed単位paired bootstrapで、raw countおよび5-feature structural baselineのMAEよりそれぞれ0.0486 [0.0468, 0.0505]、0.0362 [0.0319, 0.0404]小さかった。

さらに候補選択にも未使用の第五の独立pair-module familyへ同じmodelを適用した。全120条件のraw countは16、30,720 challengeのcertificate違反は0だったが、二成分modelのSpearmanは0、MAEは0.2238で、raw countとstructural baselineを上回らなかった。Normalized margin単独は副次的にSpearman 0.8944を示した。またpair符号だけを変える30 seedは符号座標変換で同値であり、bootstrap区間は一点へ退化した。

そこで既存5 familyを符号座標共役で監査したところ、600 network条件は192構造classへ縮約された。各gainの30 seedに対する有効class数はdense symmetric 8、sparse symmetric 2、asymmetric dense 29、feedforward non-normal 8、modular paired 1だった。続いてmodule結合絶対値をseedごとに変えた60個の有効構造networkで積則を事前登録確認した。240条件、61,440 challengeにおいてfull固定点数は常に4×4=16、全corner符号保持率のmodule積に対する最大残差は0、component certificate下界違反は0だった。保持率は0.390625–1.0に分布した。

以上は、有限外乱下で利用可能な記憶状態を評価するにはアトラクタ数だけでなくmargin profileが必要である一方、既知構造で学習した低次元の線形較正を任意の力学系へ普遍化できないことを示す。またraw seed数をtask保存対称性で割り、component別profileと積則を保持する必要がある。本結果は4次元tanh RNN、一定外乱、符号保持task、hyperbox十分条件に限定される。

キーワード: reservoir computing、multistability、robust invariant set、survivability、attractor repertoire、recurrent neural network

## 1. はじめに

Reservoir Computing Generalizedを含む近年の理論は、計算基材を特定のEcho State Networkへ限定せず、多様な入力駆動力学系とreadoutの組として計算を捉える方向を示している [1,2]。この一般化は、任意の力学系を候補reservoirとして分析し、その構造とダイナミクスをtaskへ適合させる設計問題を重要にする。

多重安定系では、アトラクタ数、吸引域、遷移障壁、条件付き安定性が記憶と状態遷移の自然な記述量になる。しかし、アトラクタが自律系に存在することと、入力または外乱下でその状態を計算資源として安全に利用できることは同じではない。Basin stabilityは摂動後にどのアトラクタへ漸近するかを測り、survivabilityは過渡軌道が望ましい領域を一度も逸脱しない確率を測る [3,4]。また、非正規結合は局所固有値だけでは捉えにくい高速な過渡増幅を生じ得る [5]。したがって、生のアトラクタ数だけで有限外乱下の利用可能性を代表させることには理論的な不足がある。

本研究の問いは次の三点である。第一に、非対角tanh RNNの各符号orthantへ任意方向有界外乱に対する計算可能な安全marginを与えられるか。第二に、raw autonomous attractor countを一致させてもrobust repertoireを分離できるか。第三に、そのmargin profileが未知seedの外乱下記憶性能をraw count、coupling、局所Jacobianよりよく予測するか。

主な貢献は以下である。

- 符号変換したtanh RNNに対し、共通境界orthant hyperboxがロバスト正不変となる一様外乱marginを導出した。
- raw count、認証count、外乱強度別robust repertoire curve、margin分布を分離する分析APIを実装した。
- count-matchedな未使用seed確認と、122,880 challengeからなる4 familyの事前登録task確認を実施した。
- 保証下界と経験的点予測を分離し、coupling、局所Jacobian、固定点座標、非正規性を比較baselineとして実装した。
- 対象familyのlabelをfitから除外するfamily・seed二重holdoutで、robust repertoire二成分modelの移送性能を事前登録確認した。
- 符号座標共役の証人探索とclass監査を実装し、既存600 network条件の実効構造多様性を192 classと算定した。
- 符号共役で異なる60個の異質独立module networkで、固定点、component certificate、全corner task保持率の積則を事前登録確認した。

## 2. 関連研究

### 2.1 Reservoir computingの一般性とtask依存性

Reservoir computingの普遍近似理論は、fading memory filterや確率入力を含む設定で整備されてきた [2,6]。一方、fading memoryを持たないreservoirでも右無限時間operatorを用いることで近似可能性が得られる例が示されている [7]。これらは多重安定な基材をreservoir候補から排除しない理論的背景になるが、個々のアトラクタをtask上どの程度安全に利用できるかは与えない。

Topologyと性能の関係もtask依存である。2026年のrandom reservoir比較では、対称性の効果が予測対象の力学系によって変化した [8]。Photonic reservoirの最新研究でもsmall-world構造の利得が報告された一方、memoryと予測で最適parameterは一致しなかった [9]。従って、一つのtopologyまたは一つの静的指標から普遍的な設計則を結論しないことが重要である。

### 2.2 アトラクタ、有限時間機能、過渡安全性

Basin stabilityは非線形系の大域安定性を初期条件分布に基づき評価する [3]。Survivabilityは安全領域からの過渡逸脱を別に測る [4]。離散時間非線形系のinput-to-state stabilityは、有界入力に対する状態応答を扱う一般枠組みを与える [10]。

固定点の存在と有限時間機能も一致しない。2026年の連想記憶modelでは、平衡論的な記憶状態が消える容量超過後にもslow regionによるtransient retrievalが残り得ることが示された [11]。逆に、本研究が扱う問題は、固定点が存在していても有限外乱に対する安全性が不足し得るという補完的な側面である。

### 2.3 非正規性と神経回路

非正規な再帰結合は、臨界固有値によるdynamical slowingとは異なる高速な過渡増幅を作る [5]。2026年の非相反Wilson–Cowan network研究は、feedforward結合とcyclic結合が過渡reactivityとnoise駆動遷移を異なる形で組織することを報告した [12]。この知見を踏まえ、本研究は対称familyだけでなくasymmetricおよびfeedforward non-normal familyを外的妥当性検証へ含める。

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

## 4. 方法

### 4.1 段階的検証

研究は探索と確認を分離した。EXP-2026-007ではcoupling 0.04と0.08を比較したが、raw countも平均2.1333低下し、count-matched判定は不成立だった。この陰性結果を保持し、事後探索でraw countを保つ0.07を選んだ。

EXP-2026-008では未使用seed 501–530を用い、coupling 0.04と0.07の両群で全seedのraw countを16に一致させた。EXP-2026-009ではdiscovery seed 401–430で外乱強度とfeatureを選び、未使用seed 601–630で符号記憶予測を確認した。EXP-2026-010ではpilot seed 801–808でfamily別条件を選び、discovery seed 801–830でpredictorをfitし、未使用seed 901–930で4 familyを確認した。

EXP-2026-011のpilotは観測済みseed 801–830をfit、901–930をtestに用い、6 candidateからpooled leave-one-family-out MAE最小のrobust pairを選択した。その後、penalty、feature、baseline、閾値を固定し、他3 familyの既観測60 seedでfitしたmodelをheld-out familyの未使用seed 1201–1230へ適用した。

EXP-2026-012では、EXP-2026-011で固定したfeature、penalty、baseline、閾値を変更せず、既知4 familyの既観測seed 801–830と901–930、計960条件でfitした。候補選択にもparameter調整にも使用していない `modular_paired` familyの未使用seed 1301–1330、計120条件を一度だけ評価した。

AUDIT-2026-001では性能taskを再実行せず、EXP-2026-011/012の重み600条件を符号対角共役 \(W'=DWD\) でclass分割した。EXP-2026-013ではtask前に各gain30 classを確認した `modular_heterogeneous` のseed 1401–1430を用い、2 global gain、4外乱の240条件を一度だけ評価した。

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

Sparse familyでは共通条件に天井効果または固定点消失が生じたため、pilotで異なるcoupling gridと外乱budgetを固定した。従ってfamily間の絶対保持率をtopologyの因果効果として比較しない。

Modular paired familyの重みは \(aI_4\) と二つの独立2次元blockからなる。module間結合は0であり、全系は二つの部分系の直積へ厳密に分解する。

Modular heterogeneousでは各pairの結合絶対値をglobal gainの0.75–1.25倍からseed別に生成した。task実行前監査で各gain30 networkが30符号共役classとなり、unorderedな二module絶対値pairにも重複がないことを確認した。

### 4.3 固定点探索と符号記憶task

全16 orthantについて初期状態 \(0.9\boldsymbol{s}\) から自律系を500 step発展させ、最終残差 \(10^{-9}\) 以下かつ全時刻で符号を保つ場合に固定点を発見したとした。各固定点を4-bit符号記憶とみなした。

外乱方向 \(\boldsymbol{d}\in\{-1,1\}^4\) の全16 cornerを列挙し、\(\boldsymbol{\eta}=e\boldsymbol{d}\) を100 step一定に印加した。全時刻で元のorthant符号を保てば成功とした。一network・一外乱強度あたり256 challengeであり、EXP-2026-010とEXP-2026-011の各confirmation総数は122,880、EXP-2026-012は30,720だった。

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

AUDIT-2026-001では600 network条件が192符号共役classへ縮約された。一方、結合絶対値を変えたEXP-2026-013は各gain30 classを保ち、task retentionが0.390625–1.0へ変化する中で \(T(e)=T_1(e)T_2(e)\) を最大残差0で確認した。従ってcomponent積則は、EXP-2026-012の退化した符号seedだけに依存する現象ではない。ただしこれはmodule間結合0の零次模型であり、次に測るべき量は弱結合で生じる積則残差である。

### 6.5 生得的機能コアと可塑的余剰への含意

本研究の長期仮説は、生物の神経系には発生的に形成された必須機能coreと、個体学習に利用できるplastic reserveがあり、新規学習からcoreへ流入する負荷が安全marginを消費するというものである。大規模画像研究は構造connectomeへの多遺伝子的影響と、多尺度構造固有モードによる機能ダイナミクス制約を支持する [16,17] が、アトラクタ余剰を直接測ってはいない。必須機能 \(k\) の外乱またはfeedback loadを \(e_k\)、marginを \(\mu_k\) とすれば、

\[
\rho_k=\frac{e_k}{\mu_k}
\]

を安全負荷率候補とできる。\(\rho_k<1\) は本hyperbox familyにおける十分条件であり、\(\rho_k\ge1\) は必ず失敗する必要条件ではない。

EXP-2026-010は、同じ状態数を持つ人工回路でもmargin分布により利用可能性が異なることを示した。EXP-2026-011はさらに、要求budgetで残る状態割合と余裕総量の二成分が、fitから除外した構造の新規seedへ移送できることを示した。しかしEXP-2026-012では積構造への線形較正移送が失敗したため、回路のmodule構成を無視した普遍scoreとしては使えない。この結果は上の仮説と整合も反証もしない。遺伝的設計、発生、可塑性、生物回路を観測していないため、発生過程または学習maskを操作し、core margin、忘却、新規task獲得の因果関係を測る別研究が必要である。

### 6.6 人間規模の条件について

現段階で人間規模の処理能力に必要なnode数、アトラクタ数、reserve次元を導くことはできない。導出可能なのは、候補となる必要条件の形式である。

第一に、環境外乱budget \(e\) に対してtask-relevantな機能同値類を覆うだけの \(N_{\mathrm{rob}}(e)\) または \(S_{\mathrm{rob}}(e)\) が必要である。第二に、必須機能の安全負荷率 \(\rho_k\) を許容範囲に保つgating、抑制、疎結合、module分離が必要になり得る。第三に、未知taskを追加する能力には、既存機能を壊さず新しい機能同値類を形成できるplastic reserveと更新energyが必要である。

独立同型な \(m\) moduleがすべて同時に成功する零次模型では、局所成功率
\(r\) と全系成功率 \(q\) の間に \(q=r^m\) が成り立つ。従って目標 \(q\)
には \(r\ge q^{1/m}\) が必要である。大規模系では局所信頼度を1へ近づける
だけでなく、冗長性、誤り訂正、階層的gating、失敗相関の制御が必要になる。
これは人間脳の実測下界ではなく、独立module仮定下の構成的scale条件である。

これらは定量的な人間下界ではない。人間規模へ接続するには、task組合せ複雑度、時間尺度階層、通信bandwidth、energy、発生記述長、plastic reserve枯渇を同時に変化させるscale lawが必要である。

## 7. 限界

第一に、対象は4次元tanh RNNであり、高次元、学習済み、spiking、物理reservoirへの一般化は未確認である。第二に、外乱は100 step一定のcorner方向であり、確率的、時間変動、状態依存外乱をtaskとして評価していない。Certificate自体は任意の時変成分別外乱を扱うが、経験照合の範囲は狭い。

第三に、共通境界hyperboxは保守的であり、座標別box、zonotope、polytope、level set、viability kernelより小さい安全集合しか認証しない可能性がある。第四に、raw countは16個のorthant初期値から発見した固定点数であり、全アトラクタの完全列挙ではない。第五に、sparse familyの外乱budgetとcoupling gridは他familyと異なり、topology間の絶対性能差を因果的に比較できない。

第六に、EXP-2026-010のcouplingとlocal Jacobianに対するpooled優位は事前登録secondary endpointである。第七に、EXP-2026-011はfold内で対象familyをfitから除外したが、candidate選択には4 familyすべてのpilot成績を用いた。第八に、EXP-2026-012の第五familyではrobust-pair線形modelが外挿に失敗した。さらに30 seedは符号共役で同値なため、modular family母集団に対する区間推定ではない。第九に、normalized margin単独の第五family相関は同じconfirmation上のsecondary結果であり、独立確認されていない。第十に、AUDIT-2026-001は符号対角共役だけを扱い、node permutation、一般similarity、近似共役は未監査である。第十一に、EXP-2026-013は独立な2+2 node moduleの積則を確認したが、弱結合残差またはcomponent-aware predictorの優位を検証していない。第十二に、別task family、basin-weighted safe mass、生物学的core–reserve仮説、人間規模条件は本稿の実験から直接導かれない。

## 8. 結論

自律アトラクタ数を完全に一致させても、有限外乱下で利用できる記憶状態は異なり得る。符号orthantごとのrobust invariant hyperbox marginを用いることで、この差を保証下界と経験予測の両面から記述できた。未使用seedと4 network familyで平均marginは外乱下符号記憶保持率と高い順位相関を持った。さらにcurveの高さと面積からなるrobust pairは、familyをfitから除外した未知seed予測でraw countと多変量structural baselineより小さいMAEを示した。

しかし、candidate選択にも未使用の独立module familyでは同じ線形modelが外挿に失敗した。さらに既存600 network条件は符号共役で192 classへ縮約された。異質独立moduleでは固定点、component certificate、task保持率の積則を確認できたため、Atlasはraw seed数とglobal scoreだけでなく、task保存対称性で割った有効構造class、component別profile、積則残差を保持すべきである。従って本研究の結論は「アトラクタ数から外乱budget付き・構造分解付きprofileへ評価単位を移す必要がある」までであり、「一つのglobal scoreまたは線形較正が任意の力学系へ通用する」ではない。次の焦点は弱結合残差、確率外乱、高次元系、より豊かなset表現へ一般化し、最終的にprofileへの介入が同一予算baselineより学習、記憶、頑健性を改善するか検証することである。

## データ・コードと再現性

実験spec、seed、判定、導出済みartifact、主張台帳、実装、テストは本repositoryに保存した。主要記録はEXP-2026-008からEXP-2026-013とAUDIT-2026-001である。EXP-2026-011 confirmationはsource/test manifest SHA-256 `b022077a3279917d02805a021048382f0b50f33387283d5c900a82b3ff9d0fcd`、EXP-2026-012は `30f1f7a11953dc6d8a5d1a7415ba8e12c311e718691475976883aa477295187d` で事前固定した。EXP-2026-012実行前には全145テストが通過し、branch coverageは88%だった。初回出力転送失敗後、結果未観測のまま同一code・同一seedを決定論的にreplayしてartifactを回収した。EXP-2026-013は全165 testと11 subtest、branch coverage 88%を確認し、manifest `85fb1caea1ebfb68db4e4f1ffd722534a9d6265a9db800e82d14df40739555fa` を固定して一度だけ実行した。

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
