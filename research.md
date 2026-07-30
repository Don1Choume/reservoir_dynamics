# アトラクタ指向リザバー・ダイナミクス研究計画

最終更新: 2026-07-30

位置づけ: 調査報告、仮説、ツール設計、実験計画を統合した研究ロードマップ

## 0. 結論

本研究の中心仮説は有望である。ただし、単純な「アトラクタの種類や数が多いほど、計算能力と記憶能力が高い」という形では成立しない可能性が高い。研究対象は、少なくとも次の三つの計算モードに分ける必要がある。

1. **入力駆動型のフィルタ計算**
   同じ入力履歴に対する応答の再現性、条件付き安定性、入力履歴の分離性、減衰時間、IPC/TIPCが重要である。初期値ごとに異なる応答へ分岐する多重安定性は、文脈として利用されない限り性能低下の原因になり得る。
2. **自律・閉ループ生成**
   学習対象の不変集合を再構成し、そのリアプノフスペクトル、幾何、長期統計、吸引域を維持する能力が重要である。ここでは複数の周期・準周期・カオスアトラクタを共存させることが直接的な能力になる。
3. **連想記憶・状態遷移計算**
   アトラクタを記憶内容、吸引域を誤り訂正範囲、吸引域境界や遷移障壁を切替コストとみなせる。生のアトラクタ数ではなく、到達可能性、識別可能性、吸引域の大きさ、ノイズ耐性、遷移時間の課題整合性が実効容量を決める。

したがって、本研究で作るべきものは単一の「アトラクタ数測定器」ではない。任意の力学系に対し、有限の観測・計算予算の下で、

- どのような安定・準安定な振る舞いが発見されたか
- どの初期条件・入力・摂動から到達するか
- どれほど区別でき、どれほど長く情報を保持するか
- どの入力・制御・ノイズで相互遷移するか
- それらが対象タスクに必要な時間尺度と演算へどう対応するか
- どのパラメータ変更が因果的に性能を改善するか

を、信頼区間と実験履歴付きで返す **Attractor and Capacity Atlas** と、その指標を用いる多目的調整器である。

## 1. 既存の前提をどう修正するか

### 1.1 Reservoir Computing Generalized の正確な射程

Kubotaらの *Reservoir Computing Generalized* [R1] は、従来はリザバー状態に要求されていた再現性を、最終出力へ移す考え方を提示した。一般形を

\[
x_{t+1}=F_\theta(x_t,u_t,\xi_t), \qquad
z_t=h(x_t), \qquad
\hat y_t=R_\phi(z_t,z_{t-1},\ldots)
\]

と書く。状態 \(x_t\) が初期値、内部位相、あるいは自律ダイナミクスに依存しても、時間不変変換を実現する読み出し \(R_\phi\) がタスク非関連成分を除去し、入力履歴に対して再現可能な出力を作れれば計算に利用できる、という拡張である。

これは「任意の力学系が、任意のタスクを、そのまま高性能に解ける」という定理ではない。少なくとも次の条件は別に検証する必要がある。

- 必要な入力履歴が観測 \(z_t\) またはその履歴から識別できること
- 使用する読み出しクラスが必要な不変変換を表現できること
- 有限データ、観測ノイズ、有限精度でも変換を学習できること
- リザバーから読み出しへ移した計算量を含めて比較すること
- 未知入力、未知初期値、パラメータドリフトに対して出力が再現すること

[R1] は2026年7月30日時点で査読済み版を確認できないプレプリントである。本研究では重要な作業仮説として扱うが、確立済みの一般則とは区別する。

### 1.2 ESP、Fading Memory、分離性、可観測性は別物である

| 概念 | 主に保証するもの | 単独では保証しないもの |
|---|---|---|
| Echo State Property | 同一の左無限入力履歴に対する状態の一意性、初期値依存の消失 | タスクに必要な情報が残ること |
| Fading Memory | 遠い過去の入力の影響が連続的に減衰すること | 長期記憶、高性能、頑健性 |
| Separation | タスク上異なる入力履歴を異なる表現へ写すこと | 不要な変動を無視すること |
| Generalization | タスク上同等な入力を近い表現へまとめること | 力学系としての安定性 |
| Observability | 内部状態や必要量を観測履歴から識別できること | 有限データで容易に学習できること |
| Readout approximation | 読み出しクラスが目標写像を近似できること | リザバー自身の寄与、学習効率 |

\(\rho(W)<1\) は、非線形かつ入力駆動された一般のRNNに対するESPの必要十分条件ではない [R3]。本研究では自律系のスペクトル半径だけでなく、同一入力を与えた複製間の収束、入力条件付きリアプノフ指数、generalized synchronization、bubblingへの安定余裕を測る。

ESNの普遍近似結果は、主に有界入力上のfading-memory filterの表現可能性を述べる [R4, R5]。また、Sugiuraらはリザバー自身のfading memoryを普遍性の必須条件としない十分条件を示している [R6]。いずれも、有限データでの学習、任意のランダム個体、任意タスクでの高性能を保証するものではない。

### 1.3 Edge of Chaos は設計原理ではなく検証対象である

臨界点近傍で能力が最大になる例はあるが [R10]、短期記憶の最適点がedge of chaosより前に現れる例 [R11]、edge of chaosを一般則とみなせない例 [R12]、アトラクタ再構成で十分に負の条件付きリアプノフ指数が必要になる例 [R13] がある。2026年の研究でも、最適点を条件付きリアプノフ指数ゼロへ単純に同一視できないこと [R26]、強く収縮的で弱非線形な領域でも高い記憶性能を得られること [R27] が報告されている。

従って、本研究は「最大リアプノフ指数をゼロに近づける」単目的最適化を採用しない。調整対象は次のPareto問題である。

- 条件付き安定余裕
- 適切な長さの記憶
- 入力履歴の分離性と有効次元
- 課題整合的なIPC/TIPC
- アトラクタの到達可能性と識別可能性
- 吸引域と遷移障壁
- ノイズ、初期値、パラメータずれへの頑健性
- 読み出しのサイズ、学習データ量、推論コスト

## 2. 研究対象の操作的定義

### 2.1 三種類のアトラクタを混同しない

#### A. 自律アトラクタ

\(u_t=0\) または一定入力の下での自由系 \(F_\theta\) が持つ不変集合である。固定点、極限周期、トーラス、カオスアトラクタなどを含む。ネットワーク固有のダイナミクスを理解するには重要だが、入力駆動タスクの性能を単独で決めるとは限らない。

#### B. 入力駆動アトラクタ

時間依存入力の下では系は非自律である。対象は通常の自律アトラクタよりも、pullback attractor、random attractor、入力条件付き応答集合、generalized synchronization manifoldである。同じ入力に対して複数の安定応答が残る度合いはecho index [R14] として扱える。

#### C. 学習後の閉ループアトラクタ

予測出力を入力へ戻す系では、学習により新たな閉ループ力学系が生まれる。評価すべきなのは一点ごとの予測誤差だけではなく、学習対象と同じ不変測度、リアプノフ指数、周期軌道、吸引域、遷移構造を再構成しているかである [R13, R15, R16]。

### 2.2 「アトラクタ数」は有限実験では下限である

任意の高次元非線形系について、隠れアトラクタを含む全アトラクタを有限時間で列挙することは一般には期待できない。ツールが報告する値は、次の条件付きの **発見済み個数の下限** とする。

\[
\widehat N_{\mathrm{attr}}
=
\widehat N_{\mathrm{attr}}
(\mathcal X_0,\mu_0,\mathcal U,T,\varepsilon,B)
\]

ここで、\(\mathcal X_0\) は初期状態探索領域、\(\mu_0\) は初期条件分布、\(\mathcal U\) は入力・摂動モデル、\(T\) は観測時間、\(\varepsilon\) は同一性判定許容誤差、\(B\) は計算予算である。

結果には必ず、探索領域、初期条件数、収束判定、積分器、刻み幅、過渡除去時間、クラスタ判定、乱数seed、未収束率を添える。厳密な個数を装わない。

### 2.3 実効アトラクタ容量

生の個数に代えて、探索分布の下での吸引確率 \(p_i\) から

\[
N_{\mathrm{eff}}
=
\exp\left(-\sum_i p_i\log p_i\right)
\]

を定義する。さらに、各アトラクタの観測可能性 \(o_i\)、頑健性 \(r_i\)、他との識別性 \(d_i\)、到達可能性 \(a_i\) を別軸として保持する。

\[
\mathcal A_{\mathrm{profile}}
=
\left(
N_{\mathrm{eff}},
\{p_i\},
\{o_i\},
\{r_i\},
\{d_i\},
\{a_i\},
P_{ij}(\tau),
\mathrm{MFPT}_{ij}
\right)
\]

初期段階では恣意的な重みで一つのスコアへ潰さない。課題ごとに、どの軸が性能を説明するかを階層モデルと因果介入で検証する。

### 2.4 「遷移しやすさ」には摂動モデルが必要である

遷移しやすさは力学系だけの属性ではない。何を入力し、どの方向に、どの強度で、どれだけの時間加えるかによって変わる。

- **決定論的制御:** 最小制御エネルギー、最短切替時間、吸引域境界までの距離
- **確率的摂動:** 平均初回通過時間（MFPT）、committor、反応流束、準ポテンシャル、最小作用経路
- **入力駆動:** 入力クラスに条件付けた遷移確率 \(P_{ij}(\tau\mid u)\)、成功率、誤遷移率
- **有限精度計算:** 量子化、丸め、積分誤差に対する遷移率

従って、遷移グラフの各辺には摂動モデルと信頼区間を持たせる。

## 3. 中心研究課題と反証可能な仮説

### RQ1: アトラクタ構造は、どの計算モードで能力を説明するか

**H1:** 入力駆動型の短期フィルタ課題では、自律アトラクタ数より、低いecho index、条件付き収縮率、入力分離性、課題整合IPC/TIPCが性能を強く説明する。多重安定性は、初期状態が明示的な文脈変数として使われる場合にのみ利益になる。

反証条件: 自律アトラクタ数だけで、条件付き安定性と容量を統制した後も未知タスク性能を一貫して予測できる。

### RQ2: 多重安定性の何が連想記憶能力を決めるか

**H2:** 連想記憶容量は生のアトラクタ数よりも、\(N_{\mathrm{eff}}\)、吸引域の均衡、識別距離、ノイズ保持曲線、偽アトラクタ率によって説明される。

反証条件: 吸引域と識別性を統制しても、単純な個数が想起性能の最良予測子であり続ける。

### RQ3: 遷移時間尺度とタスク時間尺度は整合するか

**H3:** metastable state間のimplied timescale、MFPT、条件付き遷移確率が、タスクの必要遅延や状態保持時間と整合するとき性能が最大になる。

反証条件: 遷移時間尺度を大きく変えても、必要遅延を変えたタスク間で性能ピークが移動しない。

### RQ4: アトラクタ指標を用いた調整は性能を因果的に改善するか

**H4:** 容量、吸引域、遷移、条件付き安定性を目的とする多目的調整は、edge-of-chaos調整、スペクトル半径調整、タスク損失だけの調整より、未知seed、未知タスク、未知力学系で高い性能と頑健性を示す。

反証条件: 同じ探索予算の下で、アトラクタ指標を追加しても汎化性能、頑健性、サンプル効率のいずれも改善しない。

### RQ5: Generalized RC の利得とコストは何か

**H5:** 状態がESPを満たさない系でも、時間不変な履歴読み出しにより有用な出力再現性を得られる。ただし必要な読み出し複雑度は、タスク非関連な自律自由度、観測不足、条件付き不安定性とともに増える。

比較では次を同時に報告する。

- 状態のreplica consistencyと出力のreplica consistency
- 読み出しのパラメータ数、FLOPs、履歴長
- 学習データ量と学習時間
- ノイズ、初期値、パラメータずれに対する性能
- リザバーを固定・シャッフル・除去した対照

## 4. Attractor and Capacity Atlas の分析体系

### 4.1 六つの分析軸

| 軸 | 主要量 | 主な意味 |
|---|---|---|
| 条件付き安定性 | 条件付きLyapunov指数、replica距離、echo index、bubbling margin | 同一入力に対する再現性 |
| レパートリー | 発見アトラクタ、\(N_{\mathrm{eff}}\)、metastable state、偽アトラクタ率 | 保持・生成できる状態の多様性 |
| 吸引域 | basin stability、basin entropy、境界感度、保持曲線 | 想起の許容範囲と誤り訂正 |
| 遷移 | 遷移行列、implied timescale、MFPT、committor、制御エネルギー | 切替の容易さと時間尺度 |
| 幾何・混合 | Lyapunov spectrum、次元、PSD、自己相関、RQA、持続ホモロジー | 軌道の不安定性、形、長期統計 |
| 計算容量 | memory curve、IPC/TIPC、有効rank、分離性、タスク性能 | 入力履歴に対する演算資源 |

### 4.2 手法の適用範囲

| 手法 | 得られるもの | 適用上の注意 |
|---|---|---|
| 多初期値シミュレーションと軌道signature clustering | 発見アトラクタ、吸引確率、代表軌道 | 隠れアトラクタと長い過渡を見逃す |
| Lyapunov / FTLE / CLV | 不安定方向、時間尺度、局所感度 | 有限時間、数値誤差、接空間実装に依存 |
| Recurrence plot / RQA [A1, A2] | 周期性、再帰、determinism、laminarity | 埋め込みと閾値依存。単独分類に使わない |
| Koopman / EDMD / DMDc [A3, A4] | 周波数、減衰モード、入力応答 | 有限辞書と観測関数に依存 |
| Ulam / transfer operator [A5] | almost-invariant set、確率的遷移 | 次元の呪い。低次元またはlatent空間向け |
| MSM / VAMPnet [A6, A7] | metastable state、implied timescale | Markov性、lag time、latent表現を検証する |
| Transition Path Theory [A8, A9] | committor、反応流束、主要遷移路 | 遷移データが希少な場合は不確実性が大きい |
| Basin stability / entropy [A10, A11] | 吸引確率、境界の不確実性 | 初期条件分布を明記しない値は比較不能 |
| Persistent homology / Mapper [A12] | ループ、連結成分、形状差 | トポロジーはダイナミクスや安定性の証明ではない |
| Conley–Morse graph [A13, A14] | 不変集合と大域遷移の検証 | 主に低次元・粗視化された系で実用的 |
| Quasipotential / gMAM [A15] | ノイズ誘起遷移障壁、最小作用路 | 小雑音近似と低い有効次元が必要 |

### 4.3 次元別の解析戦略

- **2～3次元:** 位相図、分岐継続、吸引域境界、Ulam、Conley–Morse、準ポテンシャルまで実施する。
- **有効次元10程度まで:** 局所断面、latent transfer operator、FTLE、持続ホモロジー、限定的なConley解析を組み合わせる。
- **数百～数万次元:** top-\(k\) Lyapunov指数、Jacobian-vector product、replica test、疎な再帰解析、latent MSM/Koopman、Monte Carlo basin stabilityを使う。全相空間の格子化は行わない。
- **観測しかできないblack box:** delay embedding、RQA、SINDy、Koopman、operator inference、equation-free continuationを候補とし、観測可能性を先に評価する。

### 4.4 アトラクタ同一性の判定

最終状態のユークリッド距離だけでクラスタリングしない。各軌道について、次の複合signatureを作る。

\[
s_i =
[
\text{mean/variance},
\text{PSD peaks},
\text{ACF},
\text{RQA},
\lambda_{1:k},
\text{Koopman frequencies},
\text{persistent features},
\text{invariant-measure distance}
]
\]

複数seed・複数観測窓でsignatureの安定性を検証した後にクラスタリングする。クラスタ間は再初期化、摂動、長時間延長で再検証する。UMAPやt-SNEは探索用表示に限定し、アトラクタの存在証明には使わない。

## 5. ツール設計

### 5.1 設計原則

1. **モデル非依存:** 離散時間、連続時間、確率系、観測black boxを共通adapterで扱う。
2. **条件を成果物に含める:** 初期条件分布、入力、積分器、探索予算を結果から切り離さない。
3. **不確実性を第一級データにする:** 個数、吸引確率、遷移率、容量に信頼区間を付ける。
4. **探索と検証を分離する:** 微分可能proxyによる高速調整後、非微分な分岐・多初期値解析で認証する。
5. **一つの万能指標を作らない:** 多目的profileとPareto frontを基本出力とする。
6. **同一予算で比較する:** 調整法、読み出し、baselineの探索回数と総計算量を揃える。
7. **再現可能性:** 不変な実験spec、seed manifest、環境lock、データ系譜を保存する。

### 5.2 共通インターフェース

```python
next_state = step(state, input_value, parameters, random_key)
derivative = vector_field(time, state, input_value, parameters)
observations = evolve_black_box(
    initial_state_distribution,
    input_protocol,
    parameters,
    horizon,
)
```

内部状態を持つ実装でも、入力specと結果artifactは不変オブジェクトとして扱う。ユーザー入力、範囲、solver設定はPydantic等で検証する。

### 5.3 モジュール構成

```text
reservoir_dynamics/
  systems/        # discrete / continuous / stochastic / black-box adapters
  simulation/     # integrators, batching, transient removal, replica runs
  stability/      # Lyapunov, FTLE, CLV, consistency, echo index
  atlas/          # trajectory signatures, clustering, basin estimation
  transitions/    # MSM, MFPT, committor, controlled switching
  capacity/       # memory curve, IPC, TIPC, effective rank
  topology/       # persistence and optional Conley-Morse adapters
  tuning/         # multi-objective search and differentiable proxies
  benchmarks/     # canonical systems, RC tasks, baselines
  reporting/      # figures, provenance, comparison tables
  schemas/        # immutable experiment and artifact definitions
```

ファイルは機能・ドメイン単位で小さく保ち、計算backend固有の処理をcore interfaceから分離する。

### 5.4 技術選択

**主要実装:** Python、JAX、Equinox、Diffraxを第一候補とする。

- `jax.lax.scan` とvectorizationで多数の軌道・seedを並列化できる。
- JVP/VJPで高次元系のtop Lyapunov解析と勾配ベース調整を共通化できる。
- Diffraxで常微分方程式、確率微分方程式、adjointを扱える。

**厳密・非微分解析の補助:** JuliaのDynamicalSystems.jl、Attractors.jl、BifurcationKit.jlをsidecarとして利用する。

- Pythonへ全機能を再実装せず、分岐継続や吸引域解析の検証に限定する。
- Juliaを必須依存にはせず、同じspecとartifact schemaを介して結果を交換する。

**データ駆動解析:** PySINDy、PyKoopmanまたはdeeptime、GUDHIまたはgiotto-tda、PyRQAを評価する。

依存導入前に、ライセンス、更新状況、GPU/自動微分対応、数値再現性を小規模benchmarkで比較する。MVPではJAX系、NumPy/SciPy、scikit-learn、Pydanticに絞り、重い解析はoptional extraとする。

### 5.5 可視化

標準reportには以下を含める。

- 分岐図と継続曲線
- 相図、Poincaré断面、代表軌道
- 吸引域断面、未分類領域、推定誤差
- Lyapunov spectrumと条件付き安定余裕
- recurrence plotとRQA
- 持続diagram
- アトラクタ／metastable state遷移グラフ
- committorまたは切替成功率の地図
- IPC/TIPCの遅延次数・非線形次数heatmap
- 性能、頑健性、コストのPareto front

次元削減表示には、元空間の距離や近傍をどの程度保存したかを併記する。

## 6. 調整戦略

### 6.1 調整対象

- recurrent weightのスペクトル半径だけでなく、特異値、非正規性、疎性
- leak rate、時定数分布、遅延
- 入力gain、bias、フィードバックgain
- 興奮・抑制比と符号制約
- modularity、small-world性、次数分布、motif
- ノード非線形、局所分岐パラメータ
- 観測ノードと読み出し履歴
- noise強度、量子化、物理系の制御可能パラメータ

ネットワーク構造の効果を調べる際は、次数列、重み分布、状態次元、入力数、訓練予算を保ったdegree-preserving rewiringを対照に使う。

### 6.2 二段階最適化

#### 段階1: 微分可能proxyによる探索

\[
\mathcal L =
\mathcal L_{\mathrm{task}}
+ \alpha \mathcal L_{\mathrm{consistency}}
+ \beta \mathcal L_{\mathrm{basin\ margin}}
+ \gamma \mathcal L_{\mathrm{transition}}
+ \delta \mathcal L_{\mathrm{capacity}}
+ \eta \mathcal L_{\mathrm{cost}}
\]

proxy候補は、有限時間のreplica距離、soft minimum basin margin、短時間遷移成功率、top Lyapunov penalty、課題整合容量、観測rankである。

アトラクタ個数、分岐点、クラスタlabelを直接autodiffしない。これらは不連続で、有限時間判定のartifactを最適化する危険がある。

#### 段階2: 非微分な認証

上位候補に対し、多初期値・長時間軌道、solver感度、分岐継続、basin bootstrap、未知摂動、未知seedで検証する。proxyが改善しても認証指標が改善しなければ、そのproxyを棄却または再較正する。

### 6.3 最終的な設計問題

単一性能の最大化ではなく、

\[
\max_\theta
\left[
\text{task utility},
\text{robustness},
\text{effective repertoire},
\text{transition alignment}
\right],
\qquad
\min_\theta
\left[
\text{readout complexity},
\text{energy},
\text{failure rate}
\right]
\]

としてPareto frontを求める。用途別に次のprofileを選ぶ。

- filter profile: 一意応答、適度な記憶、入力分離
- associative profile: 大きく均衡した吸引域、低い偽想起率、制御可能な切替
- generator profile: 目標不変測度と安定性、不要アトラクタの抑制
- adaptive profile: metastable stateの時間尺度整合、頑健な遷移

## 7. 実験計画

### E0. 解析器のground-truth検証

**目的:** ツール自身の誤検出、solver依存、有限時間biasを定量化する。

**系:** logistic map、Hénon map、Duffing系、Lorenz系、Rössler系、必要に応じてChua回路。

**検証項目:**

- 既知の固定点、周期倍化、カオス領域の回収
- Lyapunov指数と分岐点の誤差
- 初期条件数、観測時間、刻み幅に対する発見率
- basin stabilityのbinomial / bootstrap信頼区間
- 軌道signatureの偽分割と偽統合
- ノイズと部分観測での劣化

**合格基準:** 設定した精度範囲で既知の定性的相図を再現し、未分類率と誤分類率を報告できること。

### E1. RNNで「アトラクタ数」仮説を因果分解する

**目的:** 同一規模のRNNで、レパートリー、吸引域、遷移障壁、条件付き安定性のどれが各計算モードに効くかを切り分ける。

**介入:**

- bias、feedback、modularity、非正規性、leak、入力gainを系統的に変更
- degree-preserving rewiringでネットワーク統計を可能な限り保持
- 状態次元、読み出しクラス、訓練データ、探索予算を一致

**課題:**

- delayed recall、delayed XOR / parity
- IPC/TIPC
- NARMA10/30。ただし定義、入力範囲、divergence処理を固定する [R25]
- Mackey–Glass、Lorenz、Hénonの予測
- 複数アトラクタのcue-based associative recall
- 指定アトラクタへの制御切替

**主要解析:** 媒介分析を用いて、構造介入 \(\rightarrow\) アトラクタprofile \(\rightarrow\) 性能という経路を評価する。性能だけを相関させない。

### E2. 駆動・自律・閉ループを同一個体で比較する

**目的:** 自律アトラクタの指標が、入力駆動応答や学習後の閉ループアトラクタをどの程度予測するかを調べる。

同じリザバーについて、

1. 無入力または定常入力で自律atlasを作る。
2. 共通入力でreplica consistency、echo index、条件付きLyapunov指数を測る。
3. teacher forcingでcapacityとone-step predictionを測る。
4. 閉ループ化し、不変測度、Lyapunov spectrum、climate統計、不要アトラクタを測る。

**判定:** 三モード間で共通する予測因子と、モード固有の予測因子を分離する。

### E3. アトラクタ指向調整器の有効性

**目的:** H4を直接検証する。

**比較法:**

- random search
- Bayesian optimization
- スペクトル半径またはedge-of-stability調整
- タスク損失のみのgradient tuning
- 提案するprofile-based multi-objective tuning

**評価:** 同一計算予算で、未知seed、未知入力分布、未知ノイズ、未知遅延、未調整の力学系familyへ外挿する。チューニングに使った課題だけで結論を出さない。

### E4. 連想記憶と多機能生成

**目的:** 多重安定性を明示的な計算資源として利用する。

Kongらの動的アトラクタ連想記憶 [R17] と、多機能RC [R18, R19] を再現baselineとする。周期だけでなく複数のカオスアトラクタを記憶対象とし、

- cue corruptionに対する想起成功率
- basin retention curveとAUC
- 偽アトラクタ・混成アトラクタ率
- 記憶数を増やしたときの容量曲線
- cue energy、切替時間、誤遷移率
- ノイズ誘起itinerancy

を測る。その後、吸引域の均衡、障壁、遷移時間尺度を直接目的にして改善する。

### E5. Generalized RC とblack-box力学系

**目的:** H5とツールのモデル非依存性を検証する。

Lorenz、Rössler、Lorenz–96など、状態が単純なESPを満たさない候補を用いる。linear readout、NVAR/NGRC、履歴MLP、GRU readoutを、総パラメータ数と学習データ量を揃えて比較する。

次に内部方程式を隠し、

- delay embedding
- SINDy
- Koopman / VAMP
- neural ODEまたはoperator inference
- equation-free continuation

でatlasを推定する。white-box結果との差から、観測不足とモデル誤差を分離する。最終段階で物理リザバーのログまたはhardware emulatorへ適用する。

## 8. 評価設計

### 8.1 Baseline

- linear delay line / FIR
- vanilla ESN
- random、small-world、modular、scale-free、非正規RNN
- NVAR / NGRC
- パラメータ数を合わせたGRU / LSTM
- SINDy、Koopman、neural ODE
- oracle integratorまたは真の方程式が利用できる場合の上限

小規模リザバーでは、結合を増やしたネットワークが常に有利とは限らないため [R24]、無結合または弱結合の対照も含める。

### 8.2 指標

**短期予測**

- NRMSE、MAE
- 有効予測時間。最大Lyapunov時間で正規化する

**長期・climate**

- invariant measure間のWasserstein距離またはMMD
- PSD、自己相関、極値統計
- Lyapunov spectrum、次元
- recurrence / persistence signature
- 既知アトラクタの回収率と不要アトラクタ率

**記憶・計算**

- delay別memory \(R^2\)
- IPC/TIPCの総量と次数・遅延別配分
- 有効rank、separation/generalization

**連想・遷移**

- cue corruption別想起率
- basin retention AUC
- MFPT、切替成功率、最小cue energy
- 誤遷移、偽想起、壊滅的吸引域縮小

**工学**

- wall-clock、peak memory、FLOPs概算
- 学習データ量、読み出しパラメータ数
- 失敗率、未収束率、数値例外率

### 8.3 統計

- 個体ごとに同じ入力、noise、splitを使うpaired design
- 最初に約30 seedのpilotを行い、効果量と分散から本実験の検出力を設計
- 単一平均だけでなくmedian、interquartile mean、95% bootstrap CI、失敗率を報告
- trajectory block bootstrapで時系列相関を保持
- seed、task、system familyを階層化したmixed-effectsまたはBayesian hierarchical model
- 多数比較にはHolm補正、分布仮定が弱い比較にはFriedman / Wilcoxonを使用
- 調整と評価のseed、タスク、パラメータ領域を分離
- 全手法のチューニング予算を一致

事前に主要評価量、除外条件、divergence処理、停止条件を登録する。NARMAは実装差で結果が大きく変わるため、式、係数、初期化、入力範囲をartifactに含める [R25]。

## 9. 開発・研究ロードマップ

### Phase 0: 定義と再現基盤（0～2か月）

**成果物**

- system adapter、experiment schema、seed manifest
- logistic、Duffing、Lorenz、ESNの最小実装
- trajectory artifactとreportの仕様
- benchmark定義書
- CI上の小規模再現テスト

**Gate 0**

- 同じspecから同じartifactを再生成できる
- solver、刻み幅、seed、過渡除去条件が全結果に記録される

**進捗 2026-07-30**

- 共通入力を複数初期状態へ与える離散時間replica simulatorを実装した。
- スカラーtanhリザバーで大域的収縮上界を再現した
  [EXP-2026-001](docs/research/experiments/EXP-2026-001.md) を記録した。
- 条件付きLyapunov指数とreplica同期を分離した
  [EXP-2026-002](docs/research/experiments/EXP-2026-002.md) を記録した。
- これは既知理論のground truthであり、原著論文の新規結果ではない。
- canonical systems、artifact schema、seed manifestが未完了のためGate 0は未通過である。
- 論文化判断は[論文化ゲート](docs/research/publication-readiness.md)で管理する。

### Phase 1: Atlas MVP（2～5か月）

**成果物**

- 多初期値探索、signature clustering
- Lyapunov、replica consistency、echo index
- basin stabilityと信頼区間
- memory curve、IPCの最小版
- canonical systemのreport

**Gate 1**

- E0を通過
- 発見個数を下限として報告し、未分類率を表示できる
- 長時間延長とsolver変更による感度を自動比較できる

### Phase 2: 因果検証（5～8か月）

**成果物**

- E1、E2のデータセットと再現notebook/report
- TIPC、RQA、transition graph
- ネットワーク介入と媒介分析

**Gate 2**

- 「どのアトラクタ量がどの計算モードを予測するか」について、少なくとも一つの反証可能な結論を得る
- 未知seedで再現しない相関は設計原理から除外する

### Phase 3: 調整器と連想記憶（8～12か月）

**成果物**

- 微分可能proxyと多目的optimizer
- 非微分認証pipeline
- E3、E4
- Pareto atlas dashboard

**Gate 3**

- 同一予算のbaselineに対し、未知条件で性能または頑健性を改善
- proxy改善と認証指標改善の関係を定量化

### Phase 4: Generalized RC とblack box（12～18か月）

**成果物**

- 履歴readoutの公平比較
- SINDy / Koopman / equation-free adapter
- 方程式を隠したE5
- 物理系またはhardware emulatorでの実証

**Gate 4**

- 状態非再現性と出力再現性を分けて評価できる
- white-boxとblack-boxの誤差要因を説明できる
- 別の力学系familyへ移植できる設計指標を一つ以上示す

## 10. 最初の8週間の実施項目

1. 本文の三モードと用語をADRとして固定する。
2. `SystemAdapter`、`ExperimentSpec`、`TrajectoryArtifact`を定義する。
3. logistic、Duffing、Lorenz、vanilla ESNを実装し、ground-truth testを先に書く。
4. 多初期値batch simulationと過渡除去を実装する。
5. 最大Lyapunov指数、replica consistency、memory curveを実装する。
6. 軌道signatureの最小版をPSD、ACF、最大Lyapunov指数で作る。
7. basin stabilityをbootstrap CI付きで実装する。
8. E0の精度・計算量曲線を作る。
9. E1のpilot用に、同一状態次元のESN群とdegree-preserving rewiringを用意する。
10. delayed recall、IPC、cue-based recallを同一評価APIに載せる。

8週間終了時の判断は「高度な可視化ができたか」ではなく、未知seedを含む再実行で、解析値と不確実性が一貫して得られるかで行う。

## 11. 想定される失敗と対策

| リスク | 何が起きるか | 対策 |
|---|---|---|
| 隠れアトラクタ | 発見数を真の個数と誤認 | 下限として報告、探索分布と予算を明示、継続法を併用 |
| 長い過渡 | metastable stateをアトラクタと誤認 | 観測時間延長、survival curve、複数窓で検証 |
| 次元の呪い | basinやtransfer operatorが破綻 | Monte Carlo、latent解析、JVP、低次元断面へ切替 |
| 部分観測 | 異なる状態を同一と誤認 | delay embedding、観測可能性評価、複数sensor比較 |
| TDAの過解釈 | 形の違いを安定性と解釈 | Lyapunov、再帰、遷移解析との複合signatureに限定 |
| autodiff artifact | 数値積分器や有限時間proxyを攻略 | solver変更、長時間・非微分認証、gradient check |
| benchmark leakage | 調整課題だけ改善 | 未知task、未知遅延、未知系familyで評価 |
| 読み出しへの能力移転 | GRCの利得をリザバー能力と誤認 | 総パラメータ、FLOPs、データ量を揃え、reservoir ablation |
| edge-of-chaos先入観 | ゼロLyapunov近傍だけ探索 | 広い安定領域とPareto探索、条件付き指数を測定 |
| 一つの万能指標 | 相反する用途を同じscoreで最適化 | profileを保持し、用途別Pareto選択 |

## 12. 論文化の単位

1. **Atlas methodology paper**
   有限予算下でのアトラクタ発見下限、不確実性、複合signature、canonical systemsでの検証。
2. **Causal reservoir design paper**
   構造介入、条件付き安定性、吸引域、遷移時間、IPC/TIPCと三計算モードの因果関係。
3. **Attractor-aware tuning paper**
   多目的proxy、非微分認証、未知タスク・未知系familyへの汎化。
4. **Generalized/physical RC paper**
   状態非再現性と出力再現性、読み出し複雑度、black-boxまたは物理系での実証。

最初の論文で「任意の力学系を最適化できる」と主張しない。まず測定の妥当性と適用限界を確立し、次に因果調整、最後に一般化へ進む。

## 13. 主要参考文献

### 13.1 Reservoir Computing、容量、普遍性

- [R1] Kubota et al. (2024), “Reservoir Computing Generalized.” arXiv:2412.12104. <https://arxiv.org/abs/2412.12104>
- [R2] Jaeger (2001), “The ‘Echo State’ Approach to Analysing and Training Recurrent Neural Networks.” GMD Report 148. <https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02>
- [R3] Yildiz, Jaeger, and Kiebel (2012), “Re-visiting the Echo State Property.” *Neural Networks*. <https://doi.org/10.1016/j.neunet.2012.07.005>
- [R4] Grigoryeva and Ortega (2018), “Echo State Networks are Universal.” *Neural Networks*. <https://doi.org/10.1016/j.neunet.2018.08.025>
- [R5] Grigoryeva and Ortega (2018), “Universal Discrete-Time Reservoir Computers with Stochastic Inputs and Linear Readouts Using Non-Homogeneous State-Affine Systems.” *JMLR* 19. <https://jmlr.org/papers/v19/18-020.html>
- [R6] Sugiura et al. (2024), “Nonessentiality of Reservoir’s Fading Memory for Universality.” *IEEE TNNLS*. <https://doi.org/10.1109/TNNLS.2023.3298013>
- [R7] Dambre et al. (2012), “Information Processing Capacity of Dynamical Systems.” *Scientific Reports* 2, 514. <https://doi.org/10.1038/srep00514>
- [R8] Kubota, Takahashi, and Nakajima (2021), “A Unifying Framework for Information Processing in Stochastically Driven Dynamical Systems.” *Physical Review Research* 3, 043135. <https://doi.org/10.1103/PhysRevResearch.3.043135>
- [R9] Ohkubo and Inubushi (2024), “Reservoir Computing with Generalized Readout Based on Generalized Synchronization.” *Scientific Reports* 14, 30918. <https://doi.org/10.1038/s41598-024-81880-3>

### 13.2 安定性、臨界性、アトラクタ再構成

- [R10] Boedecker et al. (2012), “Information Processing in Echo State Networks at the Edge of Chaos.” *Theory in Biosciences*. <https://doi.org/10.1007/s12064-011-0146-8>
- [R11] Haruna and Nakajima (2019), “Optimal Short-Term Memory Before the Edge of Chaos in Driven Random Recurrent Networks.” *Physical Review E* 100, 062312. <https://doi.org/10.1103/PhysRevE.100.062312>
- [R12] Carroll (2021), “Do Reservoir Computers Work Best at the Edge of Chaos?” *Chaos*. <https://doi.org/10.1063/5.0038163>
- [R13] Hart (2024), “Attractor Reconstruction with Reservoir Computers: The Effect of the Reservoir’s Conditional Lyapunov Exponents on Faithful Attractor Reconstruction.” *Chaos* 34, 043123. <https://doi.org/10.1063/5.0196257>
- [R14] Ceni et al. (2020), “The Echo Index and Multistability in Input-Driven Recurrent Neural Networks.” *Physica D* 412, 132609. <https://doi.org/10.1016/j.physd.2020.132609>
- [R15] Pathak et al. (2017), “Using Machine Learning to Replicate Chaotic Attractors and Calculate Lyapunov Exponents from Data.” *Chaos*. <https://doi.org/10.1063/1.5010300>
- [R16] Lu, Hunt, and Ott (2018), “Attractor Reconstruction by Machine Learning.” *Chaos*. <https://doi.org/10.1063/1.5039508>
- [R17] Kong, Brewer, and Lai (2024), “Reservoir-computing Based Associative Memory and Itinerancy for Complex Dynamical Attractors.” *Nature Communications*. <https://doi.org/10.1038/s41467-024-49190-4>
- [R18] Du et al. (2025), “Multifunctional Reservoir Computing.” *Physical Review E* 111, 035303. <https://doi.org/10.1103/PhysRevE.111.035303>
- [R19] Flynn, Tsachouridis, and Amann (2021), “Multifunctionality in a Reservoir Computer.” *Chaos*. <https://doi.org/10.1063/5.0019974>
- [R20] Zhang and Cornelius (2023), “Catch-22s of Reservoir Computing for Basin Prediction.” *Physical Review Research* 5, 033213. <https://doi.org/10.1103/PhysRevResearch.5.033213>
- [R21] Kabayama et al. (2025), “Designing Chaotic Attractors: A Semisupervised Approach.” *Physical Review E* 111, 034207. <https://doi.org/10.1103/PhysRevE.111.034207>
- [R22] Kobayashi et al. (2026), “On the Attractor in High-Dimensional Neural Network Dynamics of Reservoir Computing: A Lyapunov Analysis Viewpoint.” *Chaos* 36, 053115. <https://doi.org/10.1063/5.0315384>
- [R23] Yan et al. (2024), “Emerging Opportunities and Challenges for the Future of Reservoir Computing.” *Nature Communications* 15, 2056. <https://doi.org/10.1038/s41467-024-45187-1>
- [R24] Jaurigue et al. (2024), “Chaotic Attractor Reconstruction Using Small Reservoirs—the Influence of Topology.” arXiv:2402.16888. <https://arxiv.org/abs/2402.16888>
- [R25] Wringe, Stepney, and Trefzer (2025), “Reservoir Computing Benchmarks: A Tutorial Review and Critique.” *International Journal of Parallel, Emergent and Distributed Systems* 40(4), 313–351. <https://doi.org/10.1080/17445760.2025.2472211>
- [R26] Suetani and Parlitz (2026), “Impact of Weak Generalized Synchronization on Time Series Forecasting Using Reservoir Computers.” *Chaos* 36, 043125. <https://doi.org/10.1063/5.0283017>
- [R27] Metzner et al. (2026), “Illuminating the Black Box of Reservoir Computing.” *Scientific Reports* 16, 15500. <https://doi.org/10.1038/s41598-026-53098-y>

### 13.3 力学系・遷移・トポロジーの分析

- [A1] Eckmann, Kamphorst, and Ruelle (1987), “Recurrence Plots of Dynamical Systems.” *Europhysics Letters*. <https://doi.org/10.1209/0295-5075/4/9/004>
- [A2] Marwan et al. (2007), “Recurrence Plots for the Analysis of Complex Systems.” *Physics Reports*. <https://doi.org/10.1016/j.physrep.2006.11.001>
- [A3] Williams, Kevrekidis, and Rowley (2015), “A Data-Driven Approximation of the Koopman Operator.” *Journal of Nonlinear Science*. <https://doi.org/10.1007/s00332-015-9258-5>
- [A4] Proctor, Brunton, and Kutz (2016), “Dynamic Mode Decomposition with Control.” *SIAM Journal on Applied Dynamical Systems*. <https://doi.org/10.1137/15M1013857>
- [A5] Dellnitz and Junge (1999), “On the Approximation of Complicated Dynamical Behavior.” *SIAM Journal on Numerical Analysis*. <https://doi.org/10.1137/S0036142996313002>
- [A6] Sarich, Noé, and Schütte (2010), “On the Approximation Quality of Markov State Models.” *Multiscale Modeling & Simulation*. <https://doi.org/10.1137/090764049>
- [A7] Mardt et al. (2018), “VAMPnets for Deep Learning of Molecular Kinetics.” *Nature Communications*. <https://doi.org/10.1038/s41467-017-02388-1>
- [A8] Metzner, Schütte, and Vanden-Eijnden (2009), “Transition Path Theory for Markov Jump Processes.” *Multiscale Modeling & Simulation*. <https://doi.org/10.1137/070699500>
- [A9] E and Vanden-Eijnden (2010), “Transition-Path Theory and Path-Finding Algorithms for the Study of Rare Events.” *Annual Review of Physical Chemistry*. <https://doi.org/10.1146/annurev.physchem.040808.090412>
- [A10] Menck et al. (2013), “How Basin Stability Complements the Linear-Stability Paradigm.” *Nature Physics*. <https://doi.org/10.1038/nphys2516>
- [A11] Daza et al. (2016), “Basin Entropy: A New Tool to Analyze Uncertainty in Dynamical Systems.” *Scientific Reports*. <https://doi.org/10.1038/srep31416>
- [A12] Perea and Harer (2015), “Sliding Windows and Persistence: An Application of Topological Methods to Signal Analysis.” *Foundations of Computational Mathematics*. <https://doi.org/10.1007/s10208-014-9206-z>
- [A13] Dey, Mrozek, and Slechta (2022), “Persistence of Conley–Morse Graphs in Combinatorial Dynamical Systems.” *SIAM Journal on Applied Dynamical Systems* 21(2), 817–839. <https://doi.org/10.1137/21M143162X>
- [A14] Vieira et al. (2022), “A Pipeline for Data-Driven Analysis of Complex Dynamical Systems Using Morse Graphs.” arXiv:2202.08383. <https://arxiv.org/abs/2202.08383>
- [A15] Heymann and Vanden-Eijnden (2008), “The Geometric Minimum Action Method.” *Communications on Pure and Applied Mathematics*. <https://doi.org/10.1002/cpa.20238>
- [A16] Morr, Kuehn, and Datseris (2026), “Computing Resilience Measures in Dynamical Systems.” *Chaos* 36, 023102. <https://doi.org/10.1063/5.0303938>

## 14. 調査方法と確度

本稿は、Reservoir Computing、非線形力学、データ駆動力学系、遷移解析、トポロジカルデータ解析、数値計算基盤という六領域に分けて調査した。主張の根拠には、原著論文、査読誌、公式プロジェクト文書を優先した。レビューやプレプリントは、最先端の方向性または未確定の仮説を示す場合に限定して用いた。

特に確度に注意すべき点は次の通りである。

- GRC [R1]、小規模reservoir [R24]、Morse graph pipeline [A14] は本文執筆時点でプレプリントとして扱う。
- edge of chaos、アトラクタ数、ネットワーク構造に関する結論は系・入力・タスク依存であり、普遍則として扱わない。
- 高次元系の全アトラクタ列挙、完全な吸引域境界、厳密な遷移障壁はMVPの保証対象にしない。
- 新規ライブラリの採用可否は、今後公式文書、ライセンス、保守状況、再現benchmarkを確認して決定する。

本研究の新規性は、既存指標を並べることではない。自律、入力駆動、閉ループという異なる力学系を同じ実験規約で比較し、アトラクタのレパートリー、吸引域、遷移、条件付き安定性、IPC/TIPCを因果介入と多目的調整へ結び付ける点に置く。

## 15. 継続的な理論・証拠・実装の運用

本計画の各主張は、機械可読な
[研究主張台帳](docs/research/claims.toml)で、出典付き事実、ローカル再現、
推論、仮説に分離して管理する。数学的定義と数値推定量の境界は
[基礎指標の数学的根拠](docs/research/theory/core-metrics.md)、未解決部分と
次の識別実験は[未解決課題](docs/research/open-questions.md)へ記録する。

実装へ進めるのは、数学的定義または既存研究上の根拠が明確で、数値的な
適用限界をテストで固定できる部分である。未解決の関係を事実として実装へ
埋め込まず、反証条件付きの仮説として登録する。各実験は
[実験記録テンプレート](docs/research/experiments/TEMPLATE.md)から作成し、
成功・陰性・判定不能のすべてを台帳へ戻す。

最新研究との対応付けは
[文献監視手順](docs/research/literature-watch.md)に従い、少なくとも各実験
フェーズ開始時と論文投稿前に更新する。GRC、条件付き安定性、アトラクタ
設計のように更新が速い領域は月次確認候補とする。
