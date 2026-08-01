# アトラクタ数を越えたリザバー評価

## ロバスト・レパートリー余裕による外乱下記憶性能の予測

研究草稿 v0.2
2026年7月30日

## 要旨

リザバー計算に用いる力学系の能力を理解するうえで、アトラクタの数は直観的な指標である。しかし、自律系で同数のアトラクタを持つ二つの系が、入力、雑音、学習済みmoduleからのfeedbackを受けたときにも同数の状態を安全に利用できるとは限らない。本研究では、離散時間tanh RNNの各符号orthantについて、成分別有界外乱下でロバスト正不変となる共通境界hyperboxを構成し、その最大一様外乱余裕を導出した。外乱budget \(e\) に対して認証可能なアトラクタ数をロバスト・レパートリー \(N_{\mathrm{rob}}(e)\) と定義し、生の自律アトラクタ数から分離した。

4次元RNNを用いた段階的な検証では、まず未使用30 seedで自律アトラクタ数を全条件16に一致させたまま、認証レパートリーを分離した。次に、全16 orthantと全16一定corner外乱方向からなる符号記憶taskを導入した。単一familyの未使用30 seedでは、低外乱で認証robust fractionと保持率のSpearman相関が0.8823、高外乱で平均marginとの相関が0.9347だった。さらにdense symmetric、sparse symmetric、asymmetric dense、feedforward non-normalの4 familyを事前登録条件で確認したところ、全480 network条件のraw countが16のまま、平均marginと保持率の相関は0.8933–0.9771となった。

最後に、各foldの対象familyをfitから除外し、他3 familyだけで標準化ridge回帰をfitするfamily・seed二重holdout確認を行った。要求外乱で無次元化した平均marginと認証robust fractionの二成分modelは、新規30 seedのpooled MAE 0.0822、family別Spearman 0.8225–0.9572を示した。seed単位paired bootstrapで、raw countおよび5-feature structural baselineのMAEよりそれぞれ0.0486 [0.0468, 0.0505]、0.0362 [0.0319, 0.0404]小さかった。

以上は、有限外乱下で利用可能な記憶状態を評価するにはアトラクタ数だけでなく、外乱budgetを引数とするmargin profileが必要であることを支持する。一方、本結果は4次元tanh RNN、既知の4 family、一定外乱、符号保持task、hyperbox十分条件に限定される。candidate選択にも未使用の第五family、確率外乱、高次元系、生物回路、人間規模の必要条件への一般化は今後の課題である。

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

### 4.2 Network family

全networkは4次元、対角自己結合1.5のtanh RNNである。

| Family | 非対角構造 | Coupling gain | 外乱budget |
|---|---|---|---:|
| Dense symmetric | 全無向edge、seed別符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Sparse symmetric | 次数2の無向ring、seed別符号 | 0.04, 0.06, 0.08, 0.10 | 0.12 |
| Asymmetric dense | 全有向edge、方向別seed符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |
| Feedforward non-normal | 上三角有向edge、seed別符号 | 0.04, 0.05, 0.06, 0.07 | 0.16 |

Sparse familyでは共通条件に天井効果または固定点消失が生じたため、pilotで異なるcoupling gridと外乱budgetを固定した。従ってfamily間の絶対保持率をtopologyの因果効果として比較しない。

### 4.3 固定点探索と符号記憶task

全16 orthantについて初期状態 \(0.9\boldsymbol{s}\) から自律系を500 step発展させ、最終残差 \(10^{-9}\) 以下かつ全時刻で符号を保つ場合に固定点を発見したとした。各固定点を4-bit符号記憶とみなした。

外乱方向 \(\boldsymbol{d}\in\{-1,1\}^4\) の全16 cornerを列挙し、\(\boldsymbol{\eta}=e\boldsymbol{d}\) を100 step一定に印加した。全時刻で元のorthant符号を保てば成功とした。一network・一外乱強度あたり256 challengeであり、EXP-2026-010とEXP-2026-011の各confirmation総数は122,880だった。

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

### 4.5 統計と事前判定

Family別にSpearman順位相関とMAEを算出した。誤差差は同じseedの4 family・各4 gainを平均して一標本とし、30 seedを2,000回percentile bootstrapして95%区間を求めた。

EXP-2026-010の事前判定は、confirmation全480 networkでraw count 16、certificate下界違反0、4 familyすべてでmean marginとtask retentionのSpearmanが0.75超、pooled raw-count MAE minus margin MAEの95%区間下限が0超、の4項目である。Couplingとlocal Jacobianに対する誤差差はsecondary endpointとして事前に固定した。

EXP-2026-011では各confirmation seedの4 family・4 gainを平均して一標本とした
paired誤差差を用いた。事前判定は、raw count 16、certificate違反0、全familyで
robust-pair予測とのSpearman 0.75超、raw-count MAE minus robust-pair MAEの
95%区間下限0超、structural MAE minus robust-pair MAEの区間下限0超、の5項目
である。確認seedを観測する前にsource/test manifestを固定した。

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

## 6. 考察

### 6.1 アトラクタ数から利用可能性へ

本結果の中心は、全networkが同じ16個の自律符号固定点を持つ条件でも、外乱下保持率が大きく異なることである。Raw countは存在を数えるが、有限外乱に対する安全性を含まない。\(N_{\mathrm{rob}}(e)\) とmargin分布は、要求外乱budgetに応じて利用可能な状態を数え直す。

低外乱ではthreshold付きrobust fractionが有効であり、高外乱では連続的な平均marginがより情報を持った。EXP-2026-011では、curveの要求点での高さ \(R(e)\) と無次元化面積 \(\bar\mu/e\) の組が、いずれか単独より小さい未知seed MAEを示した。従って、分析ツールは単一scoreへ縮約せず、raw count、certified count、\(N_{\mathrm{rob}}(e)\) curve、margin分布、未認証率を保持すべきである。

### 6.2 保証と予測の分離

Hyperbox certificateは任意の時変成分別外乱に対する十分条件である。今回の一定corner taskでは、認証robust fractionは全条件で経験保持率の下界になった。ただし未認証orthantにも成功例があり、点予測としては保守的である。

従ってツールは二つの出力を分ける必要がある。第一は反例が出れば理論または実装が誤っている保証下界、第二はdataset上の相関や交差検証誤差として評価する経験予測である。この分離により、「認証不能」を「失敗」と誤解することを避けられる。

<!-- pagebreak -->

### 6.3 局所安定性と有限外乱安全性

Local Jacobianは固定点近傍の微小摂動増幅を測るが、符号境界までの有限距離とworst-case方向を直接含まない。Orthant marginは両者をhyperbox十分条件の形で含む。EXP-2026-010のpooled secondary解析でmarginがlocal Jacobianを上回ったことは、この差がtask予測に寄与する可能性を示す。

ただしfamily内では局所baselineが優位な例もあった。EXP-2026-011ではcoupling、局所Jacobian、minimum coordinate、nonnormalityを同時に含むstructural modelを事前登録baselineとし、robust pairがpooled MAEを上回った。これは線形ridgeの範囲での増分予測力であり、非線形model、family indicatorを用いた階層model、nested cross-validationとの比較は残る。

### 6.4 生得的機能コアと可塑的余剰への含意

本研究の長期仮説は、生物の神経系には発生的に形成された必須機能coreと、個体学習に利用できるplastic reserveがあり、新規学習からcoreへ流入する負荷が安全marginを消費するというものである。必須機能 \(k\) の外乱またはfeedback loadを \(e_k\)、marginを \(\mu_k\) とすれば、

\[
\rho_k=\frac{e_k}{\mu_k}
\]

を安全負荷率候補とできる。\(\rho_k<1\) は本hyperbox familyにおける十分条件であり、\(\rho_k\ge1\) は必ず失敗する必要条件ではない。

EXP-2026-010は、同じ状態数を持つ人工回路でもmargin分布により利用可能性が異なることを示した。EXP-2026-011はさらに、要求budgetで残る状態割合と余裕総量の二成分が、fitから除外した構造の新規seedへ移送できることを示した。この結果は上の仮説と整合するが、遺伝的設計、発生、可塑性、生物回路を観測していない。生物学的主張には、発生過程または学習maskを操作し、core margin、忘却、新規task獲得の因果関係を測る別研究が必要である。

### 6.5 人間規模の条件について

現段階で人間規模の処理能力に必要なnode数、アトラクタ数、reserve次元を導くことはできない。導出可能なのは、候補となる必要条件の形式である。

第一に、環境外乱budget \(e\) に対してtask-relevantな機能同値類を覆うだけの \(N_{\mathrm{rob}}(e)\) または \(S_{\mathrm{rob}}(e)\) が必要である。第二に、必須機能の安全負荷率 \(\rho_k\) を許容範囲に保つgating、抑制、疎結合、module分離が必要になり得る。第三に、未知taskを追加する能力には、既存機能を壊さず新しい機能同値類を形成できるplastic reserveと更新energyが必要である。

これらは定量的な人間下界ではない。人間規模へ接続するには、task組合せ複雑度、時間尺度階層、通信bandwidth、energy、発生記述長、plastic reserve枯渇を同時に変化させるscale lawが必要である。

## 7. 限界

第一に、対象は4次元tanh RNNであり、高次元、学習済み、spiking、物理reservoirへの一般化は未確認である。第二に、外乱は100 step一定のcorner方向であり、確率的、時間変動、状態依存外乱をtaskとして評価していない。Certificate自体は任意の時変成分別外乱を扱うが、経験照合の範囲は狭い。

第三に、共通境界hyperboxは保守的であり、座標別box、zonotope、polytope、level set、viability kernelより小さい安全集合しか認証しない可能性がある。第四に、raw countは16個のorthant初期値から発見した固定点数であり、全アトラクタの完全列挙ではない。第五に、sparse familyの外乱budgetとcoupling gridは他familyと異なり、topology間の絶対性能差を因果的に比較できない。

第六に、EXP-2026-010のcouplingとlocal Jacobianに対するpooled優位は事前登録secondary endpointである。第七に、EXP-2026-011はfold内で対象familyをfitから除外したが、candidate選択には4 familyすべてのpilot成績を用いた。candidate選択にも未使用の第五family、非線形baseline、別task family、basin-weighted safe massは未検証である。第八に、生物学的core–reserve仮説と人間規模条件は本稿の実験から直接導かれない。

## 8. 結論

自律アトラクタ数を完全に一致させても、有限外乱下で利用できる記憶状態は異なり得る。符号orthantごとのrobust invariant hyperbox marginを用いることで、この差を保証下界と経験予測の両面から記述できた。未使用seedと4 network familyで平均marginは外乱下符号記憶保持率と高い順位相関を持った。さらにcurveの高さと面積からなるrobust pairは、familyをfitから除外した未知seed予測でraw countと多変量structural baselineより小さいMAEを示した。

本研究は「アトラクタの数」から「外乱budget付きの利用可能なアトラクタprofile」へ評価単位を移す。次の焦点は、candidate選択にも未使用の第五family、確率外乱、高次元系、より豊かなset表現へ一般化し、最終的にmargin profileへの介入が同一予算baselineより学習、記憶、頑健性を改善するかを検証することである。

<!-- pagebreak -->

## データ・コードと再現性

実験spec、seed、判定、導出済みartifact、主張台帳、実装、テストは本repositoryに保存した。主要記録はEXP-2026-008、EXP-2026-009、EXP-2026-010、EXP-2026-011である。EXP-2026-011 confirmationはsource/test manifest SHA-256 `b022077a3279917d02805a021048382f0b50f33387283d5c900a82b3ff9d0fcd` で事前固定した。組版検証追加後の現行manifestは `7db1dc7182893688558eaf7cb7095a8f2ad16c2712620a49f79787df8b4d33f1` であり、全145テストと11 subtestが通過し、branch coverageは88%だった。

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
