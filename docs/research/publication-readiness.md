# 論文化ゲート

判定日: 2026-07-30  
現在の判定: **査読付き原著論文には未到達**

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
- 実装は全119テストを通過し、branch coverageは90%である。

### 陰性結果

- 負のtop CLEかつ非同期となる率が0.8以上という事前登録判定は、観測率
  0.7333のため不成立だった。
- 局所容量とshared worst容量の絶対差の95%区間下限が0.5を超えるという
  multistability penalty判定は、最大でも下限0.3284で不成立だった。
- この2件は事後的に成功へ変更せず、次の確認実験の設計根拠とする。

### 未検証

- アトラクタprofileが計算・記憶能力を予測するという中心仮説
- 生のアトラクタ数と実効レパートリーの比較
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
- robust repertoire curveの予測がcouplingと局所安定性から独立するか
- robust repertoireと符号記憶の関係が別network・task familyで維持されるか
- random matched mask、部分空間保護、EWC、replayに対するcore–reserve
  構造の増分優位
- module数、task数、reserve次元、feedback総量、energyのscale law

## 推論

現状は局所安定性、状態同期、固定readoutの機能移送性を分離できる再現可能な
研究基盤である。30 seed条件と区間推定は得られたが、単一network familyの
選択済み条件に限られる。

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

## 次回判定

以下が揃った時点で再判定する。

- 独立seedでrelative retentionの確認基準を事前登録し、再現する
- 少なくとも非直交・疎RNNを含む3 network familyで検証する
- Lymburn型global/output consistencyをbaselineとして直接比較する
- 非線形taskと、線形・二次・履歴readoutの公平比較を行う
- 自律atlas上で機能同値類を推定し、未知初期状態性能を交差検証する
- 非対角・高次元の多重安定coreで安全集合marginを推定・外部認証する
- sparse symmetric、asymmetric、non-normal familyでrobust curveを確認する
- coupling、局所Jacobian、固定点座標に対する増分予測力を交差検証する
- protected block、matched random mask、部分空間保護を同一budgetで比較する
- task逐次追加によるreserve枯渇とfeedback budgetのscale lawを推定する
