# 実装済み基礎指標の数学的根拠

この文書は、コードが計算している量と、その量からは結論できないことを分離する。
記号や推定法を変更するときは、対応する主張IDとテストも同時に更新する。

## 1. Basin stability

対応主張: `C-DYN-001`, `C-STAT-001`

力学系のフローまたは写像を \(\Phi_t\)、初期条件を採る確率測度を
\(\mu_0\)、発見済みアトラクタを \(A_i\) とする。有限観測時間 \(T\) と
分類規則 \(g_T\) の下で、実際に推定する量は

\[
S_i(T,\mu_0,g_T)
=
\Pr_{x_0\sim\mu_0}\left[g_T(\Phi_T(x_0))=i\right]
\]

である。無限時間極限の真の吸引域体積そのものではない。独立な初期条件
\(x_0^{(1)},\ldots,x_0^{(n)}\) に対して

\[
\widehat S_i
=
\frac{1}{n}\sum_{k=1}^{n}
\mathbf 1\left[g_T(\Phi_T(x_0^{(k)}))=i\right]
\]

を用いる。これは指定した \(\mu_0,T,g_T\) に条件付けたbinomial比率推定である。

### Wilson score interval

\(\hat p=k/n\)、標準正規分布の両側信頼水準 \(1-\alpha\) に対応する分位点を
\(z=z_{1-\alpha/2}\) とすると、

\[
\mathrm{center}
=
\frac{\hat p + z^2/(2n)}{1+z^2/n},
\]

\[
\mathrm{halfwidth}
=
\frac{z}{1+z^2/n}
\sqrt{
\frac{\hat p(1-\hat p)}{n}
+\frac{z^2}{4n^2}
}.
\]

実装は \([\mathrm{center}-\mathrm{halfwidth},
\mathrm{center}+\mathrm{halfwidth}]\) を \([0,1]\) に制限する。

この区間が表すのは標本化不確実性だけである。隠れアトラクタ、誤分類、
数値積分誤差、相関した初期条件は別途感度分析する。

## 2. 実効レパートリー

対応主張: `C-DYN-002`

発見済みアトラクタへの到達確率を \(p_1,\ldots,p_m\) とする。
Shannon entropy

\[
H(p)=-\sum_{i:p_i>0}p_i\log p_i
\]

の指数

\[
N_{\mathrm{eff}}=\exp(H(p))
\]

を実効レパートリーと定義する。これはorder 1のHill numberであり、

- 一つのアトラクタへ確率1で到達すると \(N_{\mathrm{eff}}=1\)
- \(m\) 個へ等確率で到達すると \(N_{\mathrm{eff}}=m\)
- 同じ生の個数でも吸引域が偏ると \(N_{\mathrm{eff}}<m\)

となる。

これは多様性の数学的集約であり、計算能力との関係は仮説である。吸引域の
大きさ、可観測性、遷移可能性を失うため、単独の最適化目的にはしない。

## 3. Replica距離

対応主張: `C-RC-001`

同一入力 \(u_{0:t}\) を与え、異なる初期状態から開始した \(M\) 個の複製を
\(x_t^{(1)},\ldots,x_t^{(M)}\in\mathbb R^d\) とする。実装は

\[
D(t)
=
\sqrt{
\frac{2}{M(M-1)d}
\sum_{1\le i<j\le M}
\left\|x_t^{(i)}-x_t^{(j)}\right\|_2^2
}
\]

を返す。座標数 \(d\) で正規化するため、状態次元の異なる実験間で単純な
距離増加だけを比較しにくくする。

十分に小さい摂動領域で

\[
D(t)\approx D(0)e^{\lambda_c t}
\]

が成り立つなら、\(\log D(t)\) の傾きは最大条件付きLyapunov指数
\(\lambda_c\) の有限時間推定と関係する。しかし飽和、ノイズ床、非線形過渡、
複数の安定応答があるため、全区間の単純回帰を指数推定として採用しない。

有限個の入力実現で \(D(t)\) が減衰したことは、その入力条件での経験的
再現性を示すにすぎず、大域的ESPの証明ではない。

## 4. 証明、数値認証、経験的指標の境界

| 出力 | 言えること | 言えないこと |
|---|---|---|
| `effective_repertoire_size` | 与えた確率分布の均衡を考慮した多様性 | 全アトラクタ数、性能 |
| `estimate_basin_stability` | 指定分布・分類規則での到達率と標本CI | 隠れアトラクタ不存在 |
| `pairwise_replica_distance_curve` | 指定入力・初期値標本での複製間距離 | 一般入力に対するESP |

今後、低次元系では分岐継続やConley–Morse解析を数値認証として追加する。
高次元系では、証明できない量を「発見済み下限」または「有限時間推定」と
明記する。

## 5. スカラーtanhリザバーの収縮上界

対応主張: `C-RC-004`

\[
x_{t+1}=\tanh(a x_t+b u_t+c)
\]

を考える。同一入力 \(u_t\) を受ける別の複製を \(\tilde x_t\) とする。
\(\tanh\) の導関数は

\[
\frac{d}{dz}\tanh(z)=\operatorname{sech}^2(z)\in(0,1]
\]

である。平均値の定理から、

\[
\begin{aligned}
|x_{t+1}-\tilde x_{t+1}|
&=
|\tanh(a x_t+b u_t+c)-\tanh(a\tilde x_t+b u_t+c)|\\
&\le |a|\,|x_t-\tilde x_t|.
\end{aligned}
\]

帰納法により、

\[
|x_t-\tilde x_t|\le |a|^t|x_0-\tilde x_0|
\]

を得る。従って \(|a|<1\) は、任意の共通入力列に対する大域的な指数収縮の
十分条件である。

重要なのは十分条件である点である。\(|a|\ge1\) でも、入力が軌道を
\(\operatorname{sech}^2(z)\ll1\) の飽和領域へ駆動すれば条件付き収縮が
起こり得る。この境界外を調べるには、解析上界だけでなく条件付きLyapunov
指数とreplica距離を併用する。

数値照合は [EXP-2026-001](../experiments/EXP-2026-001.md) に記録した。
これは既知の収縮議論の再現であり、新規定理ではない。

## 6. 局所条件付き安定性と大域的一意性

対応主張: `C-RC-005`, `C-RC-006`

スカラーtanhリザバーの局所微分は

\[
J_t
=
\left|
\frac{\partial x_{t+1}}{\partial x_t}
\right|
=
|a|(1-x_{t+1}^2)
\]

である。washout \(W\) 後の有限時間条件付きLyapunov指数を

\[
\widehat\lambda_c
=
\frac{1}{T-W}
\sum_{t=W}^{T-1}\log J_t
\]

と推定する。\(\widehat\lambda_c<0\) は、その参照応答の近傍で微小摂動が
平均的に減衰することを示す。

しかし、これは全初期状態が同じ応答へ収束することを意味しない。無入力、
biasなしでは固定点は

\[
x^\ast=\tanh(a x^\ast)
\]

を満たす。\(a>1\) では原点の傾きが1を超え、奇対称性により非零の
\(\pm x^\ast\) が現れる。それぞれの固定点で

\[
|a|(1-(x^\ast)^2)<1
\]

なら局所的には安定で、各軌道の条件付き指数は負になり得る。一方、正負の
初期状態は別の固定点へ収束するため、大域的な一意応答は存在しない。

逆に \(|a|\ge1\) でも、共通入力が状態を飽和領域
\(|x_{t+1}|\approx1\) へ駆動すれば \(J_t\ll1\) となり、特定入力に対して
同期し得る。

従って、少なくとも次の三量を分離して報告する。

1. 大域的Lipschitz上界による十分条件
2. 参照軌道に沿った条件付きLyapunov指数
3. 複数初期状態からのreplica距離またはecho index

数値結果は [EXP-2026-002](../experiments/EXP-2026-002.md) に記録した。

## 7. 多次元RNNの最大条件付き指数と線形記憶

対応主張: `C-RC-007`, `C-RC-008`, `C-RC-009`

多次元tanh RNN

\[
\boldsymbol{x}_{t+1}
=
\tanh(W\boldsymbol{x}_t+B\boldsymbol{u}_t+\boldsymbol{c})
\]

の状態Jacobianは

\[
J_t
=
D_tW,\qquad
D_t
=
\operatorname{diag}
\left(1-\boldsymbol{x}_{t+1}^{\odot2}\right)
\]

である。単位接ベクトル \(\boldsymbol{v}_t\) に対して

\[
\widetilde{\boldsymbol{v}}_{t+1}=J_t\boldsymbol{v}_t,\qquad
\boldsymbol{v}_{t+1}
=
\frac{\widetilde{\boldsymbol{v}}_{t+1}}
{\|\widetilde{\boldsymbol{v}}_{t+1}\|_2}
\]

を反復し、

\[
\widehat{\lambda}_{\max}
=
\frac{1}{T-W_0}
\sum_{t=W_0}^{T-1}
\log\|\widetilde{\boldsymbol{v}}_{t+1}\|_2
\]

を最大有限時間条件付きLyapunov指数とする。

これは一つの入力実現と一つの参照軌道に条件付けられる。負であっても、別の
吸引域に属する有限距離の初期状態が同じ応答へ収束するとは限らない。

### 線形記憶曲線

遅延 \(k\) について、学習区間で

\[
\widehat{\boldsymbol{w}}_k
=
\arg\min_{\boldsymbol{w}}
\sum_t
\left(
u_{t-k}
-\boldsymbol{w}^{\mathsf T}
[1;\boldsymbol{x}_t]
\right)^2
+\alpha\|\boldsymbol{w}_{1:}\|_2^2
\]

を解く。独立test区間における予測値を \(\widehat u_{t-k}\) とし、

\[
MC_k
=
\operatorname{corr}^2
\left(u_{t-k},\widehat u_{t-k}\right),\qquad
MC_{\mathrm{linear}}=\sum_{k=1}^{K}MC_k
\]

を報告する。

理想化した仮定では総容量はアクセス可能な独立状態次元に制約される。しかし
有限test標本では、無関係な遅延にも正の標本相関が生じるため、推定総和は
状態次元をわずかに超え得る。この超過を新しい自由度の発見と解釈しない。

`EXP-2026-003` では、次の三量を同時に報告した。

1. \(\widehat{\lambda}_{\max}\): 局所的な微小摂動の平均成長率
2. replica距離: 標本化した初期状態間の大域的再現性
3. \(MC_{\mathrm{linear}}\): 一つの参照応答から線形に読める遅延情報

これにより「局所的には高容量だが、大域的には応答が一意でない」条件を
識別できる。

数値結果は [EXP-2026-003](../experiments/EXP-2026-003.md) に記録した。

## 8. 固定readoutの初期状態間移送性

対応主張: `C-RC-010`, `C-RC-011`, `C-RC-012`, `H-RC-006`

### 8.1 局所容量と共有容量

同一入力を受けるreplicaを
\(\boldsymbol{x}^{(0)}_t,\ldots,\boldsymbol{x}^{(M-1)}_t\) とする。遅延
\(k\) ごとに参照replica \(m=0\) の学習区間だけを用いて、

\[
\widehat{\boldsymbol{w}}_k
=
\arg\min_{\boldsymbol{w}}
\sum_{t\in\mathcal I_{\mathrm{train}}}
\left(
u_{t-k}
-\boldsymbol{w}^{\mathsf T}[1;\boldsymbol{x}^{(0)}_t]
\right)^2
+\alpha\|\boldsymbol{w}_{1:}\|_2^2
\]

を求める。この重みを再学習せず、replica \(m\) のtest状態へ適用する。

\[
\widehat u^{(m)}_{t-k}
=
\widehat{\boldsymbol{w}}_k^{\mathsf T}
[1;\boldsymbol{x}^{(m)}_t].
\]

共有readout容量を

\[
C^{\mathrm{shared}}_{k,m}
=
\max\left\{
0,\,
1-
\frac{
\sum_{t\in\mathcal I_{\mathrm{test}}}
(u_{t-k}-\widehat u^{(m)}_{t-k})^2
}{
\sum_{t\in\mathcal I_{\mathrm{test}}}
(u_{t-k}-\bar u_k)^2
}
\right\}
\]

とする。相関係数二乗は予測の符号反転にも1を与えるため、初期状態間で同じ
readoutを使えるかという目的には不適切である。そこでheld-out \(R^2\) を
0で切り詰める。総容量とworst-case retentionは

\[
C^{\mathrm{shared}}_m
=
\sum_{k=1}^{K}C^{\mathrm{shared}}_{k,m},
\qquad
Q_{\mathrm{worst}}
=
\frac{
\min_m C^{\mathrm{shared}}_m
}{
C^{\mathrm{shared}}_0
}
\]

とする。分母が数値的に0なら数学上は比を定義できないため、実装は診断値0を
返す。この場合、reference自体に容量がないことを別途確認する。

局所容量は一つの応答軌道が保持する情報を測る。共有容量は、参照軌道で
学習した同一の座標系・係数が、標本化した別初期状態にも通用するかを測る。
従って、後者は全入力、全初期状態に対するESPや普遍的計算能力の証明ではない。

### 8.2 readoutが誘導する機能的商

固定readout \(h\) と評価時間集合 \(\mathcal I\) を明示し、出力擬距離を

\[
d_h(i,j)
=
\left(
\frac{1}{|\mathcal I|}
\sum_{t\in\mathcal I}
\|h(\boldsymbol{x}^{(i)}_t)-h(\boldsymbol{x}^{(j)}_t)\|_2^2
\right)^{1/2}
\]

とする。\(d_h\) は状態空間上では異なる応答間の距離を0にし得るため、
readoutにより誘導される擬距離である。厳密な関係を

\[
\boldsymbol{x}^{(i)}
\equiv_{h,\mathcal I}
\boldsymbol{x}^{(j)}
\quad\Longleftrightarrow\quad
d_h(i,j)=0
\]

とすれば、これは出力時系列の等しさから導かれる同値関係である。

アトラクタ集合 \(\mathcal A\) をこの関係でまとめた
\(\mathcal A/{\equiv_{h,\mathcal I}}\) を、task-specificな機能的商と呼ぶ。
実験では完全一致ではなく、task score \(S\) の差または \(d_h\) が許容誤差
\(\varepsilon\) 以下かを使う。ただし、この近似関係は推移的とは限らない。
有限標本から商を推定するには、擬距離に対する明示したclustering規則と、
閾値感度・標本外安定性が必要である。この推定法と一致性は未解決である。

Lymburn et al.はglobal reservoir consistencyとreadout方向のconsistencyを
分け、非整合なreservoirでも高整合な方向に入力情報が残ることを示した。
Generalized RCは、基材状態が同じ入力へ再現応答しなくても、処理済み入力を
取得して再現可能な出力を構成できる場合を示す。本節の新規候補はこの一般原理
ではなく、固定readoutのheld-out \(R^2\) を用いて、どの初期状態・アトラクタ
差が特定taskに有害かを直接診断し、atlasの商構造へ接続する点にある。

数値結果は [EXP-2026-004](../experiments/EXP-2026-004.md) に記録した。

## 9. 一次資料

- Menck et al., basin stability: <https://doi.org/10.1038/nphys2516>
- Wilson, binomial interval: <https://doi.org/10.1080/01621459.1927.10502953>
- Jost, entropy and diversity: <https://doi.org/10.1111/j.2006.0030-1299.14714.x>
- Manjunath and Jaeger, input-specific ESP: <https://doi.org/10.1162/NECO_a_00411>
- Ceni et al., echo index and multistability: <https://doi.org/10.1016/j.physd.2020.132609>
- Yildiz, Jaeger, and Kiebel, ESP条件: <https://doi.org/10.1016/j.neunet.2012.07.005>
- Jaeger, short-term memory capacity:
  <https://doi.org/10.24406/publica-fhg-291107>
- Dambre et al., information processing capacity:
  <https://doi.org/10.1038/srep00514>
- Lymburn et al., consistency in echo-state networks:
  <https://doi.org/10.1063/1.5079686>
- Kubota et al., Reservoir Computing Generalized:
  <https://arxiv.org/abs/2412.12104>
- Ohkubo and Inubushi, generalized readout:
  <https://doi.org/10.1038/s41598-024-81880-3>
