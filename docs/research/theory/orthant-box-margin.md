# 非対角tanh RNNのorthant-box margin

最終確認: 2026-07-30  
対応主張: `C-RC-018`, `C-RC-019`, `H-RC-007`, `H-BIO-005`

## 1. 目的

自律アトラクタが存在することと、学習module、noise、制御入力から来る有界外乱
の下で安全に使えることを分ける。非対角tanh RNNの符号orthantごとに、
ロバスト正不変hyperboxと最大一様外力を認証する。

## 2. systemと符号変換

\[
\boldsymbol x_{t+1}
=
\tanh(W\boldsymbol x_t+\boldsymbol\eta_t)
\]

を考える。対象とする符号patternを
\(\boldsymbol s\in\{-1,1\}^d\)、
\(D_s=\operatorname{diag}(\boldsymbol s)\) とし、

\[
\boldsymbol y=D_s\boldsymbol x,
\qquad
\widetilde W=D_sWD_s,
\qquad
\widetilde{\boldsymbol\eta}=D_s\boldsymbol\eta
\]

と変換する。tanhの奇対称性から、

\[
\boldsymbol y_{t+1}
=
\tanh(
\widetilde W\boldsymbol y_t
+
\widetilde{\boldsymbol\eta}_t
)
\]

である。対象orthant内の共通境界boxを

\[
S_s(m)=[m,1]^d,
\qquad 0<m<1
\]

とする。

## 3. 命題: robust orthant box

\(\widetilde W\) の第 \(i\) 行について、

\[
P_i
=
\sum_j\max(\widetilde W_{ij},0),
\qquad
N_i
=
\sum_j\min(\widetilde W_{ij},0)
\]

と定義する。

### 命題

外乱が

\[
\|\widetilde{\boldsymbol\eta}_t\|_\infty\le e
\]

を全時刻で満たし、

\[
e
\le
\min_i
\left[
P_i m+N_i-\operatorname{atanh}(m)
\right]
\]

なら、\(S_s(m)\) はロバスト正不変である。

### 証明

\(\boldsymbol y\in[m,1]^d\) とする。正の重みは \(y_j=m\)、負の重みは
\(y_j=1\) のとき第 \(i\) preactivationを最小化する。従って、

\[
\sum_j\widetilde W_{ij}y_j+\widetilde\eta_i
\ge
P_i m+N_i-e
\ge
\operatorname{atanh}(m).
\]

tanhの単調性から \(y_{t+1,i}\ge m\) であり、値域から
\(y_{t+1,i}\le1\) である。全座標で成立するため結論を得る。
\(\square\)

## 4. 最大一様margin

このhyperbox familyで認証できる最大一様外力を

\[
\mu_s
=
\max_{0<m<1}
\min_i
\left[
P_i m+N_i-\operatorname{atanh}(m)
\right]
\]

と定義する。

\[
g(m)=\min_i(P_i m+N_i)-\operatorname{atanh}(m)
\]

は区分的に凹である。最大候補は、

1. 各active branchで
   \(P_i-1/(1-m^2)=0\) を満たす
   \(m=\sqrt{1-1/P_i}\), \(P_i>1\)
2. 二つのaffine branchの交点
3. \(m\to0^+\) の境界

に限られる。実装はこれらを全列挙し、有限gridでも照合する。

\(\mu_s>0\) なら正の有界外乱まで認証できる。\(\mu_s\le0\) はこの共通境界
hyperbox familyで認証できないことだけを意味し、アトラクタ不存在、局所不安定、
最大robust invariant set不存在を意味しない。

## 5. 既知scalar式への縮退

### 1次元

\(W=(a)\), \(a>1\) なら \(P_1=a,N_1=0\) なので、

\[
m_*=\sqrt{1-\frac1a},
\qquad
\mu=a m_*-\operatorname{atanh}(m_*).
\]

[双安定coreのロバスト不変margin](bistable-core-margin.md)と一致する。

### 2次元対称coupling

\[
W=
\begin{pmatrix}
a&b\\
b&a
\end{pmatrix}
\]

とする。

- aligned orthant \((+,+)\) では、
  \(\mu=\eta_{\mathrm{crit}}(a+b)\)。
- opposed orthant \((+,-)\) では、
  \(\mu=\max(0,\eta_{\mathrm{crit}}(a)-b)\)。

従って同じnetwork内でも、coupling signとattractor signの整合によって
ロバストmarginが異なる。

## 6. raw repertoireとrobust repertoire curve

発見した自律アトラクタ集合を \(\mathcal A\)、各アトラクタの認証marginを
\(\mu_k\) とする。外乱budget \(e\) に対するロバストレパートリー数を

\[
N_{\mathrm{rob}}(e)
=
\sum_{k\in\mathcal A}
\mathbf 1[\mu_k\ge e]
\]

と定義する。

\[
N_{\mathrm{raw}}=|\mathcal A|=N_{\mathrm{rob}}(0)
\]

とは限らない。認証familyが保守的なら、\(\mu_k=0\) とした存在アトラクタが
あり得るためである。実装上は、

- discovered raw count
- certified-at-zero count
- \(N_{\mathrm{rob}}(e)\) curve
- margin分布
- 未認証率

を別々に返す。

初期条件分布からのbasin probabilityを \(p_k\) とすれば、外乱下の認証質量を

\[
S_{\mathrm{rob}}(e)
=
\sum_k p_k\mathbf 1[\mu_k\ge e]
\]

と定義できる。これは生の個数だけでなく、到達しやすさを含む安全レパートリー
曲線の候補である。

## 7. EXP-2026-007: 陰性結果を含む発見

4次元signed complete RNNのcouplingを0.04から0.08へ強めると、

- raw count: 16.0000から13.8667
- certified count: 16.0000から9.7333
- raw count対応差: 2.1333 [1.0667, 3.4667]
- certified count対応差: 6.2667 [5.7333, 6.7333]

となった。

certificate、safe外乱、boundary witnessの判定は成立したが、raw count一定の
事前判定は不成立だった。この結果を「count-matched分離」の成功へ変更しない。

## 8. EXP-2026-008: 未使用seed確認

探索で選んだcoupling 0.07を未使用seed 501–530へ固定すると、

- 両群のraw count: 全seedで16
- weak certified count: 16.0000
- strong certified count: 10.2000 [9.7333, 10.7333]
- certified count対応差: 5.8000 [5.2667, 6.2667]
- safe外乱下のbox保持率: 1
- 1.1倍boundary witness escape率: 1

となった。

この限定familyでは、raw autonomous attractor countを完全に一致させても、
任意方向有界外乱に対する認証レパートリーを分離できた。

## 9. EXP-2026-009: 外乱下taskへの接続

未使用seed 601–630を用い、全networkのraw autonomous countを16へ固定した。
外乱 \(e=0.08\) では

\[
\operatorname{Spearman}
\left(
\frac{N_{\mathrm{rob}}(e)}{16},
\mathrm{retention}
\right)
=0.8823
\]

となった。外乱 \(e=0.16\) ではthreshold countが粗くなり、

\[
\operatorname{Spearman}
\left(
\frac1{16}\sum_k\max(\mu_k,0),
\mathrm{retention}
\right)
=0.9347
\]

となった。両featureともdiscovery fitを固定した未知seed予測でraw-count
baselineより小さいMAEを示した。

認証orthantはすべての一定corner外乱で符号を保ち、各networkで

\[
\mathrm{retention}(e)
\ge
\frac{N_{\mathrm{rob}}(e)}{16}
\]

が成立した。ただしこれは認証対象外の成功率を上から制約せず、coupling
baselineへの増分予測力も確立しない。

## 10. EXP-2026-010: network familyを越えた確認

dense symmetric、sparse symmetric、asymmetric dense、
feedforward non-normalの4 family、未使用seed 901–930で、全networkの
raw autonomous countを16へ固定した。family別にdiscovery fitした
\(\bar\mu\) とtask retentionのSpearmanは、

\[
0.8933,\quad 0.9244,\quad 0.9557,\quad 0.9771
\]

だった。seed単位poolingしたMAE差は、

\[
\begin{aligned}
E_{\mathrm{raw}}-E_{\bar\mu}
&=0.0851\ [0.0828,0.0873],\\
E_{\mathrm{coupling}}-E_{\bar\mu}
&=0.0080\ [0.0030,0.0133],\\
E_{\mathrm{local\ Jacobian}}-E_{\bar\mu}
&=0.0051\ [0.0020,0.0085].
\end{aligned}
\]

後二つは事前登録secondary endpointである。poolした平均では有限外乱marginが
生成parameterと微小摂動baselineを上回ったが、feedforward family内では
local Jacobian MAEの方が小さかった。普遍的優位ではなく、4 familyにわたる
再現可能なprofile指標として解釈する。

## 11. 人間規模仮説への接続

必須機能 \(k\) のmargin \(\mu_k\) と、学習moduleからのfeedback load \(e_k\)
を比較し、

\[
\rho_k=\frac{e_k}{\mu_k}
\]

を機能別の安全負荷率候補とする。大規模系で必要なのは巨大な
\(N_{\mathrm{raw}}\) だけでなく、環境外乱budget \(e\) に対する
\(N_{\mathrm{rob}}(e)\) と \(S_{\mathrm{rob}}(e)\) を維持する構造である。

これは人間脳の必要条件として確立していない。今後、task数、module数、
network dimension、energyとともにこれらのcurveがどうscaleするかを測る。

## 12. 適用限界

- hyperboxと成分別一様外乱に限定した十分条件である。
- 共通境界 \(m\) は座標別境界より保守的になり得る。
- 4次元の4構成familyで確認したが、高次元、学習済み、spiking、物理系では
  未確認である。
- raw countは指定初期値からの固定点発見数で、全アトラクタ列挙保証ではない。
- 符号保持taskは確認したが、readout性能、記憶容量、cue routingは未検証で
  ある。
- coupling 0.07は発見実験後に選ばれ、EXP-008は同じ生成分布内のseed確認で
  ある。
- EXP-009は別seedで事前登録確認したが、coupling baselineへの増分優位は
  未確立だった。EXP-010のpooled secondary解析では優位を得たが、family内で
  一貫した支配ではない。
