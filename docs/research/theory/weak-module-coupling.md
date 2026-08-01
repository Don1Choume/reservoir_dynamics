# 弱いモジュール間結合と積則残差

更新日: 2026-08-01

## 1. 目的

独立moduleで成立する固定点・certificate・有限時間taskの積則を、module間結合
を加えた系の零次基準として使う。ここでは、積則が厳密でなくなった後にも
残る保証を導き、結合強度と積則残差を同じ分析profileに保存する。

## 2. 設定

二つの2次元moduleからなるblock diagonal行列を \(W_0\) とし、

\[
W_\varepsilon=W_0+E_\varepsilon,
\qquad \|E_\varepsilon\|_\infty=\varepsilon
\]

とする。各行に高々一つのcross-module edgeを置く実験では、edge絶対値を
\(\varepsilon\) とすればこの等号が厳密に成立する。\(\varepsilon=0\) では
`modular-factorization.md` の直積系へ戻る。

## 3. 座標別境界を持つorthant rectangle

符号 \(s\in\{-1,1\}^d\) と下側境界
\(b=(b_1,\ldots,b_d)\in(0,1)^d\) に対し、

\[
B_s(b)=\{x: b_i\le s_ix_i\le1\ \text{for all }i\}
\]

を考える。\(D_s=\operatorname{diag}(s)\)、
\(\widetilde W=D_sWD_s\) とすると、行 \(i\) の最悪入力余裕は

\[
m_i(b;W,s)=
\sum_j\left[
\max(\widetilde W_{ij},0)b_j+
\min(\widetilde W_{ij},0)
\right]-\operatorname{atanh}(b_i)
\]

である。従って

\[
M(b;W,s)=\min_i m_i(b;W,s)
\]

と置けば、\(M\ge e\) は任意の時変成分別外乱
\(\|\eta_t\|_\infty\le e\) に対する \(B_s(b)\) のロバスト正不変性の十分条件
である。これは全座標へ一つの境界を課す既存certificateを、固定された
座標別境界へ一般化したものである。

## 4. 結合normによるmargin損失上界

同じ \(s,b\) に対して \(W_0\) と \(W_0+E\) を比較する。各成分の寄与は、
係数の符号が変わる場合を含めても

\[
\min_{y_j\in[b_j,1]}(\widetilde W_{0,ij}+\widetilde E_{ij})y_j
\ge
\min_{y_j\in[b_j,1]}\widetilde W_{0,ij}y_j-|E_{ij}|
\]

を満たす。行和を取り、さらに行の最小を取れば、

\[
\boxed{
M(b;W_0+E,s)\ge M(b;W_0,s)-\|E\|_\infty
}
\]

を得る。従って独立moduleの直積rectangleがmargin
\(M_{\mathrm{component}}\) を持つなら、

\[
M_{\mathrm{component}}\ge e+\varepsilon
\]

は弱結合後にも同じrectangleが外乱 \(e\) に耐えるための、符号patternに
依存しない十分条件である。実際のcross-edge符号を使って第3節のmarginを
再計算した `transported rectangle certificate` は、このnorm-shifted下界
以上のorthantを認証できる。

## 5. task積則残差

独立moduleの有限時間保持率を \(T_1(e),T_2(e)\)、結合後の全系保持率を
\(T_\varepsilon(e)\) とし、

\[
\Delta_T(\varepsilon,e)=
T_\varepsilon(e)-T_1(e)T_2(e)
\]

を積則残差と定義する。\(\Delta_T(0,e)=0\) は実装sanity checkである。
有限個のorthant・corner・時刻を列挙するtaskでは、\(T_\varepsilon\) は一般に
\(\varepsilon\) の滑らかな関数ではなく、challengeの失敗閾値で階段状に変わる。
従って小さい \(\varepsilon\) に対する微分係数を主要量にせず、絶対残差、
符号付き残差、初回非零結合、certificate保持率を報告する。

`transported rectangle certified fraction` を \(R_{\mathrm{rect}}\)、
norm損失だけで残る割合を \(R_{\mathrm{shift}}\) とすれば、初期固定点が対応
rectangle内にある条件の下で

\[
T_\varepsilon(e)\ge R_{\mathrm{rect}}(e)
\ge R_{\mathrm{shift}}(e)
\]

が成立する。これは積則残差の符号や大きさを直接制限しないが、弱結合下で
失われていない機能レパートリーの保証下界を与える。

## 6. 証拠境界

上界はtanhの単調性、状態範囲 \([-1,1]^d\)、成分別有界外乱だけを用いる。
確率外乱、共通外乱、時間変動する結合、周期・カオスアトラクタ、学習による
重み更新にはそのまま適用しない。また、task保持率と結合normの経験関係は
上の不等式からは導かれず、独立seedで確認する必要がある。
