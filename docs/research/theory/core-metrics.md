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

## 6. 一次資料

- Menck et al., basin stability: <https://doi.org/10.1038/nphys2516>
- Wilson, binomial interval: <https://doi.org/10.1080/01621459.1927.10502953>
- Jost, entropy and diversity: <https://doi.org/10.1111/j.2006.0030-1299.14714.x>
- Manjunath and Jaeger, input-specific ESP: <https://doi.org/10.1162/NECO_a_00411>
- Ceni et al., echo index and multistability: <https://doi.org/10.1016/j.physd.2020.132609>
- Yildiz, Jaeger, and Kiebel, ESP条件: <https://doi.org/10.1016/j.neunet.2012.07.005>
