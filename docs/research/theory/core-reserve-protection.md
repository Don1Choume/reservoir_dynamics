# 機能coreと可塑reserveの保護条件

最終確認: 2026-07-30  
対応主張: `C-RC-013`, `C-RC-014`, `H-BIO-004`, `H-BIO-005`

## 1. 目的

生得的または発生初期に獲得した機能coreを保ちながら、新しい機能を追加できる
ための最小構造条件を定式化する。

本節の結果は「脳が実際にこのblock構造を使う」という生物学的証明ではない。
可塑的余剰が存在し得ることの構成的な十分条件と、coreへのfeedback漏洩を
監視するための数値certificateである。

## 2. 既存研究との対応

McClellandらのComplementary Learning Systemsは、既存の構造化された知識へ
新規経験を直接高速学習すると干渉が生じるため、海馬系の高速学習と新皮質系の
緩徐学習を相補的に使う枠組みを示した。

- McClelland, McNaughton, and O'Reilly (1995):
  <https://doi.org/10.1037/0033-295X.102.3.419>

DunckerらはRNNの既存task dynamicsを活動で定義した部分空間内に保存し、
干渉する新task dynamicsを直交部分空間へ向ける学習則を示した。

- Duncker et al. (2020):
  <https://proceedings.neurips.cc/paper/2020/hash/a576eafbce762079f7d1f77fca1c5cc2-Abstract.html>

FarajtabarらのOrthogonal Gradient Descentも、既存出力を変えないparameter
部分空間へ更新を射影する。

- Farajtabar et al. (2020):
  <https://proceedings.mlr.press/v108/farajtabar20a.html>

従って、「別moduleまたはnull spaceで学習すれば干渉を抑えられる」という
一般原理は既知である。本研究ではこれを新規原理とは主張せず、アトラクタ、
固定readout retention、plastic reserveを同一の力学系certificateへ接続する。

## 3. block力学系

状態を機能core \(\boldsymbol c_t\in\mathbb R^{d_c}\) と可塑reserve
\(\boldsymbol r_t\in\mathbb R^{d_r}\) に分ける。

\[
\begin{aligned}
\boldsymbol c_{t+1}
&=
\phi_c(
W_{cc}\boldsymbol c_t
+W_{cr}\boldsymbol r_t
+B_c\boldsymbol u_t
+\boldsymbol b_c
),\\
\boldsymbol r_{t+1}
&=
\phi_r(
W_{rc}\boldsymbol c_t
+W_{rr}\boldsymbol r_t
+B_r\boldsymbol u_t
+\boldsymbol b_r
).
\end{aligned}
\]

学習前後で \(W_{cc},W_{cr},B_c,\boldsymbol b_c\) を固定し、
\(W_{rc},W_{rr},B_r,\boldsymbol b_r\) だけを更新する操作を
reserve-only更新と呼ぶ。

## 4. 命題: core摂動上界

\(\phi_c\) は無限大normについて1-Lipschitzとする。tanhの成分適用はこの
条件を満たす。学習前後のcore距離とreserve距離を

\[
D_t=\|\boldsymbol c_t-\widetilde{\boldsymbol c}_t\|_\infty,
\qquad
R_t=\|\boldsymbol r_t-\widetilde{\boldsymbol r}_t\|_\infty
\]

とする。また

\[
L_c=\|W_{cc}\|_\infty<1,
\qquad
L_f=\|W_{cr}\|_\infty
\]

とする。同じ入力を与え、reserve差が \(R_t\le\overline R\) を満たすなら、

\[
D_{t+1}
\le
L_cD_t+L_f\overline R
\]

であり、

\[
D_t
\le
L_c^tD_0
+
L_f\overline R
\frac{1-L_c^t}{1-L_c}
\]

を得る。

### 証明

tanhの1-Lipschitz性と誘導normの劣乗法性から、

\[
\begin{aligned}
D_{t+1}
&\le
\|
W_{cc}
(\boldsymbol c_t-\widetilde{\boldsymbol c}_t)
+
W_{cr}
(\boldsymbol r_t-\widetilde{\boldsymbol r}_t)
\|_\infty\\
&\le
L_cD_t+L_fR_t\\
&\le
L_cD_t+L_f\overline R.
\end{aligned}
\]

この漸化式を反復すれば結論を得る。

## 5. 系: 厳密保護と出力保持

\(W_{cr}=0\) かつ \(D_0=0\) なら、reserveのparameterと軌道がどれだけ
変化しても

\[
D_t=0
\]

が全時刻で成り立つ。従ってcoreだけを読む任意の固定readout
\(h_c(\boldsymbol c_t)\) も厳密に不変である。

core readoutが \(L_h\)-Lipschitzなら、有限feedback時にも

\[
\|
h_c(\boldsymbol c_t)
-h_c(\widetilde{\boldsymbol c}_t)
\|
\le L_hD_t
\]

を得る。

## 6. アトラクタの解釈

\(W_{cr}=0\) のとき、全系はcoreをbase、reserveをfiberとするskew-productに
なる。coreへの射影 \(\pi_c\) は

\[
\pi_c\circ F_\theta
=
f_c\circ\pi_c
\]

を満たし、reserve parameter \(\theta\) に依存しない。

reserve学習により全状態空間のアトラクタが新生、分裂、消失しても、core因子
上の力学とcore readoutの機能同値類は変わらない。これは、

- raw attractorの個数は増え得る
- core機能の商集合は保存される
- novel readoutに対する機能同値類は増え得る

という三つを同時に満たす構成例である。

従ってplastic reserveは「未使用アトラクタの個数」ではなく、core因子を保った
ままfiber側で新しいtask-relevantな同値類を作れる能力として定義する方がよい。

## 7. 複数reserve moduleへの拡張

coreへ \(J\) 個のreserve moduleがfeedbackする場合、

\[
D_{t+1}
\le
L_cD_t
+
\sum_{j=1}^{J}L_{f,j}\overline R_j.
\]

従って、許容core偏差を \(\varepsilon\) とするworst-case certificateは

\[
\sum_{j=1}^{J}L_{f,j}\overline R_j
\le
(1-L_c)\varepsilon
\]

である。

これは人間規模の必要条件ではなく、このmodel familyにおける十分条件である。
ただしscaleを増やしてもcoreを保つには、module数とともにcoreへ流入する
誘導normの総和を無制限に増やせないことを示す。疎結合、正規化、gating、
neuromodulationを定量化する最初の設計予測になる。

## 8. 適用限界

- \(L_c<1\) は大域収縮を使う強い十分条件であり、多重安定coreを含まない。
- 上界は符号相殺とtanh飽和を使わないため保守的である。
- reserve-only maskが生物学的にどの程度厳密かは未検証である。
- coreとreserveの境界を既知としており、実データからのmodule同定を扱わない。
- 新規task学習に必要なreserve次元またはenergyの下界はまだ導いていない。

数値照合は [EXP-2026-005](../experiments/EXP-2026-005.md) に記録する。

多重安定coreでは \(L_c<1\) を使えない。双安定scalar tanh系に対する
ロバスト正不変集合とsaddle-node forcing marginへの拡張は、
[双安定coreのロバスト不変margin](bistable-core-margin.md)と
[EXP-2026-006](../experiments/EXP-2026-006.md)に記録する。
