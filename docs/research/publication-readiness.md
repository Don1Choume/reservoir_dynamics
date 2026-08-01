# 論文化ゲート

判定日: 2026-08-01  
現在の判定: **限定的な原著論文の草稿開始可**

## 証拠

### 出典付き事実

- basin stability、replica consistency、収縮写像の距離上界には既存理論がある。
- `EXP-2026-001` は、その既知上界を本ツールで再現した。

### ローカル再現

- スカラーtanhリザバーで、21時刻すべてにおいて観測replica距離が
  \(D_t\le |a|^tD_0\) を満たした。
- `EXP-2026-002` で、負の条件付きLyapunov指数と非同期replicaの共存、
  および大域収縮条件外での入力同期を再現した。
- `EXP-2026-003` で多次元tanh RNNへ拡張し、top条件付き指数、replica同期、
  線形記憶容量を36条件で同時測定した。
- 非同期12条件中9条件で負のtop条件付き指数が得られ、局所容量と大域的
  利用可能性を分離する必要性を確認した。
- 強入力による同期回復と、tanh飽和に整合する記憶低下が同じsweepで現れた。
- `EXP-2026-004` で6条件を30対応seedへ拡張し、percentile bootstrap
  95%区間を報告した。
- 強入力による同期率改善は \(a=1.2,1.5\) の両方で平均0.7333となり、
  95%区間下限はそれぞれ0.5667だった。shared worst容量も改善した。
- 強収縮条件 `(0.6,0.1)` は比較条件 `(0.9,0.5)` より局所線形記憶容量が
  平均0.7356、95%区間 [0.6305, 0.8467] 高かった。
- 状態同期率0.7333でも固定readout retentionが0.9962の条件と、同期率0で
  retentionが0.0892の条件が得られた。
- `EXP-2026-005` でcore–reserve block系を実装し、reserve-only更新に対する
  core偏差上界を導出した。zero-feedbackでは全30 seedでcore偏差0、
  core retention 1となった。
- reserve-only更新はnovel容量を平均3.5503増やし、同じ候補数と可塑parameter
  枠数で独立最適化したcore更新対照よりnovel容量1.8759、core retention
  0.9715だけ高かった。
- feedback gain 0.02、0.05、0.1でcore retentionが0.9726、0.9000、
  0.7978へ低下し、全軌道が決定論的上界を満たした。
- `EXP-2026-006` でscalar双安定coreの最大対称forcing margin
  \(\eta_{\mathrm{crit}}=am_*-\operatorname{atanh}(m_*)\)、
  \(m_*=\sqrt{1-1/a}\) を導出した。
- \(a=1.2,1.5,2.0\) の全条件で、臨界比未満の一定最悪外力は認証区間を
  保持し、臨界比超過ではsaddle-node tippingを起こした。
- cue形成reserveの30 seed評価では、臨界比0.5と0.9のcertified core
  retentionが1、比1.1と1.5の反対cue retentionが0だった。
- 臨界比0.9でも無外力basin全体の保持率は0.8693であり、attractorへの
  漸近所属と外乱下survivabilityを経験的に分離した。
- `EXP-2026-007` ではcoupling 0.08でraw countも低下し、count-matchedという
  事前判定は不成立だった。この陰性結果を保持して確認条件を分離した。
- `EXP-2026-008` の未使用30 seedでは、4次元非対角tanh RNN二群がともに
  autonomous count 16を保つ一方、orthant-box認証countは16.0対10.2となり、
  対応差95%区間は [5.2667, 6.2667] だった。
- 全認証orthantのsafe外乱保持率とcertificate超過boundary witness率は1だった。
- `EXP-2026-009` の未使用30 seedでは全120条件のraw countが16のまま、
  低外乱のrobust fractionと符号記憶保持率のSpearmanが0.8823、高外乱の
  mean marginとのSpearmanが0.9347だった。
- discovery fitを固定した未知seed予測MAEはraw-count baselineより
  0.0554 [0.0474, 0.0647]、0.0846 [0.0742, 0.0955] 小さかった。
- 61,440 challengeでcertificate下界違反は0だった。
- `EXP-2026-010` の未使用30 seed、4 network familyでは全480条件の
  raw countが16のまま、mean marginと符号記憶保持率のSpearmanが
  0.8933–0.9771だった。
- 122,880 challengeでcertificate下界違反は0だった。
- seed単位pooled予測ではmargin MAEがraw countより0.0851
  [0.0828, 0.0873]、couplingより0.0080 [0.0030, 0.0133]、worst local
  Jacobianより0.0051 [0.0020, 0.0085] 小さかった。後二つはsecondary
  endpointである。
- `EXP-2026-011` では各foldの対象familyをfitから除外し、他3 familyでfitした
  robust pairを未使用seed 1201–1230へ適用した。family別Spearmanは
  0.8225–0.9572、pooled MAEは0.0822だった。
- raw countと5-feature structural baselineに対するseed単位MAE改善区間は
  [0.0468, 0.0505] と [0.0319, 0.0404] で、事前登録5判定はすべて成立した。
- `AUDIT-2026-001` では既存5 family・2 gain・30 seedの600条件を符号対角
  共役で監査し、192有効classへ縮約した。`modular_paired` はgainごとに
  1 class、dense symmetricとfeedforward nonnormalは各8 class、sparse
  symmetricは各2 class、asymmetric denseは各29 classだった。
- `EXP-2026-013` では結合絶対値を変えた `modular_heterogeneous` を導入し、
  各gain30 class、計60有効構造を確認した。2 moduleの固定点数と符号記憶
  保持率の積則は全240条件で成立し、61,440 challengeにおけるtask積残差は
  0、保持率範囲は0.390625–1.0だった。
- componentごとの積box certificateは全条件で成立した。全座標へ共通境界を
  課すglobal certificateはcomponent margin以下であり、評価した4閾値では
  certified fractionの差が0だったが、一般の等号は主張しない。
- `EXP-2026-013` 実行前に全165テストと11 subtestを通過し、branch coverageは
  88%だった。
- `EXP-2026-014` では未使用30 seed、2 internal gain、6 cross strength、4外乱の
  1,440点でraw count 16を保ったまま、平均絶対task積残差が0から0.0831へ
  単調非減少した。strength 0.04の非零残差率は0.65、最大絶対残差は0.375だった。
- transported rectangleとnorm-shifted certificateは368,640 challengeでtask
  保持率の下界を保ち、実際のcross-edge符号を使う前者はstrength 0.04で平均
  0.5250を認証し、normだけの後者0.4406より保守性を緩和した。
- `EXP-2026-014` 実行前に全171テストと11 subtestを通過し、branch coverageは
  88%だった。

### 陰性結果

- 負のtop CLEかつ非同期となる率が0.8以上という事前登録判定は、観測率
  0.7333のため不成立だった。
- 局所容量とshared worst容量の絶対差の95%区間下限が0.5を超えるという
  multistability penalty判定は、最大でも下限0.3284で不成立だった。
- この2件は事後的に成功へ変更せず、次の確認実験の設計根拠とする。
- `EXP-2026-012` ではcandidate選択にも未使用の `modular_paired` familyで
  robust-pairのSpearmanが0、MAEが0.2238となり、raw countとstructural
  baselineに対する主要優位も不成立だった。
- 同familyの30 seedはtask-preservingな符号座標変換で同値であり、
  seed bootstrap区間が一点へ退化した。従って独立構造標本30とは扱わない。

### 未検証

- アトラクタprofileが符号保持以外の計算・記憶能力も予測するか
- 非対称結合、module size違いを含む複数modular family間の実効
  レパートリー比較
- 遷移時間尺度とタスク時間尺度の整合
- アトラクタ指向調整のbaselineに対する優位性
- `EXP-2026-004` の所見が非直交、疎、学習済み、spiking、物理reservoirで
  維持されるか
- shared-readout retentionが既存のglobal/output consistency profileを
  超える増分予測力を持つか
- 線形遅延記憶以外のtaskと、非線形・履歴readoutへの一般化
- task-specificな機能的アトラクタ商の数学的整備とatlas上での安定推定
- 生得的機能コアとplastic reserveの操作定義が新規学習と忘却を予測するか
- 高次元・非対角・非normalな多重安定coreでの安全集合margin推定
- component-awareな摂動modelが弱結合modular familyへ外挿するか
- robust repertoireがstochastic外乱、cue、readout、逐次学習taskを予測するか
- random matched mask、部分空間保護、EWC、replayに対するcore–reserve
  構造の増分優位
- module数、task数、reserve次元、feedback総量、energyのscale law

## 推論

現状は局所安定性、状態同期、固定readoutの機能移送性に加え、符号共役で
割った有効構造多様性と、独立moduleに対するexactなcomponent積則を扱える
再現可能な研究基盤である。30 seed条件、区間推定、fold内family holdout、
単一modular family内の60有効構造は得られたが、弱結合module、candidate
選択にも未使用の別task、任意の力学系には一般化していない。

Lymburnらはreservoir全体のconsistencyとreadout方向のconsistencyを既に
分離しており、Generalized RCも非再現な基材応答から再現可能出力を得る原理を
提示している。従って、「状態が非同期でも計算できる」という一般原理だけでは
新規性にならない。

原著候補は、固定readoutのheld-out \(R^2\) による初期状態間移送診断を、
自律アトラクタatlasのtask-specificな機能的商へ接続する方法論である。
今回の二つの非同期regimeはその予備証拠だが、既存consistency指標に対する
増分妥当性が未検証なので、まだ原著論文を開始しない。生得的機能コアと
plastic reserveも、現状は検証可能な研究計画またはposition paperの段階で
ある。

`EXP-2026-005` は「機能coreを変えず新規容量を追加できる」という構成的
十分性を確立した。ただしblock-triangular系の不変性と直交部分空間による
干渉抑制の一般原理は既知であり、導出した上界もLipschitz不等式の直接適用
である。この結果単独では方法論的新規性にならない。論文候補へ進めるには、
多重安定coreのbasin保護certificate、機能的商、matched random mask比較の
少なくとも一つで既存法を超える必要がある。

`EXP-2026-006` は上記の多重安定core課題をscalar系で部分的に満たし、
「basin内にいること」と「学習feedback下で安全なこと」を定量的に分離した。
しかし閉形式はscalar saddle-nodeからの直接導出で、reserve形成も有限候補
からのparameter選択である。高次元でのcertificate、実際の学習則、同一budget
baseline比較のいずれもないため、現時点の判定は原著論文未到達のままとする。

`EXP-2026-008` は未使用seedでraw countを完全に一致させ、認証レパートリーを
分離したため、中心仮説の一部に初めてcount-matchedな機構証拠を与えた。しかし
interval hyperboxという既知の直接手法、一つの4次元生成分布、task性能なし
という制約が残る。これは論文の核候補だが、単独では原著論文未到達とする。

`EXP-2026-009` はこの核を未知seedの外乱下符号記憶taskへ接続した。raw count
一定でもrobust curveとmargin分布が性能差を識別し、certificateが経験保持率の
下界になることを確認したため、「アトラクタ数だけでは利用可能な記憶状態数を
表せない」という限定命題には論文候補となる証拠がある。一方、生成parameter
であるcouplingも同程度に予測し、robust featureのcouplingに対する誤差差区間は
0を含んだ。別family、局所安定性baseline、別taskがないため、現時点では
技術報告またはpreprintの核であり、査読付き原著論文の開始判定は変更しない。

`EXP-2026-010` は、事前登録した4 network familyと未使用30 seedでこの関係を
再現し、poolしたsecondary解析でcouplingとlocal Jacobian baselineも上回った。
これにより「生のアトラクタ数ではなく、有限外乱に対するmargin profileが
利用可能な記憶レパートリーを表す」という限定命題は、原著論文の中心仮説に
置ける。

`EXP-2026-011` は各foldの対象familyをfitから除外し、他3 familyでfitした
robust pairを新規seedへ適用した。事前登録5判定はすべて成立し、raw countと
5-feature structural baselineに対するseed単位MAE改善区間下限は0を上回った。
このため「family固有の単回帰だけで成立する」という代替説明を弱めた。

ただしcandidate選択には4 familyすべてのpilot成績を用いた。sparse familyは
別の外乱budgetを用い、confirmation MAEも最大だった。従って「marginが
あらゆる局所指標を常に支配する」「任意の力学系へ一般化する」とは主張しない。
`EXP-2026-012` は、この警告を実際の反例として確定した。raw count一定と
certificate下界は未知familyでも維持されたが、特定の二特徴量線形較正は
積構造へ外挿しなかった。従って草稿は陰性結果を含めて更新するが、普遍的
predictorとしての投稿判断は見送る。

`AUDIT-2026-001` は、符号seed数を独立構造数とみなす擬似反復を定量化した。
`EXP-2026-013` はその監査結果を受け、task前に符号共役classを構造gateとして
固定し、異質な独立moduleで固定点数、certificate、task保持率の積則を確認した。
これにより完全隔離系のcomponent分解は検証済みとなったが、これは零結合の
exact baselineである。`EXP-2026-014` は対称二bridgeの弱結合で、raw countを
変えずに積則残差が結合normとともに増えることと、移送rectangle保証がtask
下界を保つことを確認した。従って次の判別力ある検証は、非対称bridgeまたは
module size差へ残差表現を外挿し、global baselineに対する増分妥当性を測ること
である。

日本語草稿は
[アトラクタ数を越えたリザバー評価](../papers/robust-repertoire-memory-ja.md)
として作成した。

## 原著論文へ進む最小条件

次のいずれかを満たした段階で論文原稿を開始する。

1. **方法論的新規性**  
   アトラクタ発見下限または複合signatureについて、既存法より誤分類、
   計算量、不確実性較正のいずれかを改善する。
2. **新しい経験則**  
   複数のRNN familyと未知seedで、アトラクタprofileの特定成分が性能を
   再現可能に予測し、単純なスペクトル半径やedge指標を上回る。
3. **因果的設計結果**  
   profileへの介入が、同一探索予算のbaselineより未知条件で性能または
   頑健性を改善する。
4. **理論結果**  
   条件付きアトラクタ構造、可観測性、読み出し複雑度の間に新しい上界・
   下界・十分条件を導出し、数値実験で適用範囲を示す。

`EXP-2026-010` と `EXP-2026-011` により条件2を限定的に満たした。4 familyと
未知seedでmargin profileの性能予測を再現し、family holdout fitでもraw count、
多変量structural baselineに対する事前登録優位を得たため、原稿を更新する。

## 投稿前の必須追加条件

草稿は開始するが、投稿前に次を満たす。

- 完全隔離と対称弱結合で得たcomponent残差表現を、非対称bridgeまたはmodule
  size違いへ外挿し、global feature baselineと事前登録比較する
- time-varying stochastic外乱でsurvival curveと保証の保守性を測る
- family、coupling、local Jacobian、固定点座標を含む非線形baselineを
  nested cross-validationで比較する
- hyperbox以外のset表現を一つ以上実装し、保守性と計算量を比較する
- manuscriptの全数値をartifactから自動再生成する

次の研究段階としては、さらに以下を継続する。

- Lymburn型global/output consistencyをbaselineとして直接比較する
- 非線形taskと、線形・二次・履歴readoutの公平比較を行う
- protected block、matched random mask、部分空間保護を同一budgetで比較する
- task逐次追加によるreserve枯渇とfeedback budgetのscale lawを推定する
