# 方向別component結合余裕と外挿可能なprofile

更新日: 2026-08-02

## 1. 目的

異なる大きさのmoduleを非対称bridgeで結合したとき、独立moduleのrobust
repertoireから全系の安全性と有限時間taskを記述する。理論的に保証できる
certificate chainと、データでのみ評価できるtask predictorを分離する。

## 2. 設定

二module離散時間系を

\[
\begin{aligned}
x^+ &= \tanh(Ax+By+\eta_x),\\
y^+ &= \tanh(Cx+Dy+\eta_y)
\end{aligned}
\]

とする。module次元はそれぞれ \(p,q\) で、\(p=q\)、\(B=C^\top\) は仮定
しない。各moduleのorthant rectangleと孤立時marginを
\(M_x(s_x),M_y(s_y)\)、外乱budgetを \(e\) とする。

## 3. 方向別流入量

bridgeが各target moduleへ与え得る最大入力は

\[
L_x=\|B\|_\infty,\qquad L_y=\|C\|_\infty
\]

である。`weak-module-coupling.md` の行別摂動評価をmoduleごとに適用すると、

\[
M_x^{\mathrm{coupled}}\ge M_x-L_x,\qquad
M_y^{\mathrm{coupled}}\ge M_y-L_y
\]

を得る。従って

\[
M_x\ge e+L_x,\qquad M_y\ge e+L_y
\]

は直積rectangleを弱結合後にも保つ十分条件である。単一global budget
\(L=\max(L_x,L_y)\) を両moduleから引く条件はこれより保守的であり、orthant
集合上の認証率をそれぞれ \(R_{\mathrm{dir}},R_{\mathrm{global}}\) とすれば

\[
R_{\mathrm{dir}}(e)\ge R_{\mathrm{global}}(e)
\]

となる。

## 4. 実bridgeを使うtransported certificate

各moduleの境界を連結した座標別rectangleに対し、実際の \(B,C\) の符号を
保持して全行marginを再計算する。認証率を \(R_{\mathrm{rect}}\) とすると、
方向別norm条件が成立するorthantは必ずtransported条件も満たすため、

\[
\boxed{
T_{\mathrm{coupled}}(e)
\ge R_{\mathrm{rect}}(e)
\ge R_{\mathrm{dir}}(e)
\ge R_{\mathrm{global}}(e)
}
\]

を得る。最初の不等式には、結合後固定点が対応rectangle内にあり、task外乱が
成分別budget \(e\) 内であることが必要である。

## 5. factorized task基準

有限個の方向code \(k=1,\ldots,K\) をmodule間で揃え、孤立moduleの保持率を
\(T_{x,k},T_{y,k}\) とする。零結合の全系保持率は

\[
T_0(e)=\frac{1}{K}\sum_{k=1}^K T_{x,k}(e)T_{y,k}(e)
\]

である。方向codeごとに積を取ってから平均する点が、単純な周辺平均の積と
異なる。結合後残差を

\[
\Delta_T=T_{\mathrm{coupled}}-T_0
\]

とする。

## 6. component-aware predictor

理論は \(\Delta_T\) の符号や大きさを決めない。そこで、結合後task結果を
特徴へ漏らさず、次を保持する。

- global profile: 次元、raw attractor率、common certificate率、平均margin、
  off-diagonal norm、nonnormality、最大bridge norm。
- component profile: \(T_0\)、孤立certificate積、\(R_{\mathrm{dir}}\)、
  \(R_{\mathrm{rect}}\)、方向別slackの平均・最小、load不均衡、size不均衡。

global profileだけのridgeをbaselineとし、同じglobal特徴へcomponent特徴を追加
したnested ridgeを比較する。小さいmoduleでfitし、seedとmodule sizeを分離した
大きいmoduleへ係数を固定して適用する。これにより「component分解が既知構造の
再記述に留まらず外挿情報を持つか」を判定する。

## 7. 既存研究との対応

- Dashkovskiy, Rüffer, WirthのISS small-gain理論は、subsystem間依存をgain
  matrixで表し、small-gain条件から全系Lyapunov関数を構成する。本稿の
  \((L_x,L_y)\) はその有限rectangle向けの単純な方向別摂動量である。
  <https://doi.org/10.1137/090746483>
- compositional neural certificateは、小系で得たISS certificateを類似構造の
  大系へ再利用する方向を示す。本実験は学習certificateではなく、陽なtanh
  rectangleでsize外挿を試す。
  <https://proceedings.mlr.press/v211/zhang23a.html>
- multitask RNNでdynamical motifがtask間で再利用され、限定した入力weightの
  学習で新taskへ速く移れることが報告されている。これはcomponent再利用の
  生物・計算論的動機であり、本稿の積則を直接証明しない。
  <https://doi.org/10.1038/s41593-024-01668-6>
- modular attractor networkには中間的modularityでbasinと収束が改善する例が
  あるため、「分離が強いほど常に良い」とは仮定しない。
  <https://arxiv.org/abs/1101.5853>

## 8. 証拠境界

certificate chainはtanhの単調性、hypercube状態範囲、固定bridge、成分別有界
外乱に対する十分条件である。task predictorの優位、確率外乱、学習中の重み
変化、連続・周期・カオスattractor、人間脳への外挿は理論から従わない。

