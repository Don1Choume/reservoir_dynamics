# 独立モジュール型リザバーの積分解

更新日: 2026-07-30

## 1. 目的

遺伝的・発生的に固定される局所回路と、学習可能な余剰自由度を同じ模型で
扱う前段として、まず相互作用しないモジュールの極限を厳密に理解する。
この文書では、ブロック対角な再帰重みを持つ離散時間tanh力学系について、
固定点、アトラクタ数、ロバストレパートリ、有限時間符号記憶taskが積分解
する条件を導く。

ここで示す結果は「脳が独立モジュールの直積である」という主張ではない。
モジュール間結合を後から導入するときの、解析可能な零次模型である。

## 2. 設定

状態を二つの部分系へ分け、

\[
x=(x^{(1)},x^{(2)}),\qquad
W=\begin{pmatrix}W_1&0\\0&W_2\end{pmatrix}
\]

とする。成分ごとのtanhを用いる更新写像は

\[
F(x;u)=\tanh(Wx+u)
=\left(
\tanh(W_1x^{(1)}+u^{(1)}),
\tanh(W_2x^{(2)}+u^{(2)})
\right)
\]

である。したがって

\[
F=F_1\times F_2
\]

が成り立つ。以下では、初期状態、アトラクタ符号、一定外乱も同じ直積に
従うものとする。

EXP-2026-012の4次元 `modular_paired` familyは

\[
W=aI_4+g\,\operatorname{blockdiag}
\left(
s_1\begin{pmatrix}0&1\\1&0\end{pmatrix},
s_2\begin{pmatrix}0&1\\1&0\end{pmatrix}
\right),\quad s_1,s_2\in\{-1,+1\}
\]

である。既定値は \(a=1.5\)、\(g\in\{0.04,0.05,0.06,0.07\}\)
とする。

## 3. 固定点とアトラクタ数

### 命題1: 固定点集合の直積

\[
\operatorname{Fix}(F)=
\operatorname{Fix}(F_1)\times\operatorname{Fix}(F_2)
\]

が成り立つ。

証明は直接的である。\(F(x)=x\) は
\(F_1(x^{(1)})=x^{(1)}\) かつ
\(F_2(x^{(2)})=x^{(2)}\) と同値である。

したがって、数え上げ対象が孤立固定点であれば、

\[
N_{\mathrm{fix}}=N_{\mathrm{fix},1}N_{\mathrm{fix},2}
\]

となる。同様に、吸引領域と周期軌道を含むアトラクタを、直積写像の下で
同じ時間基準により定義できる場合、部分系アトラクタの直積が全系の
アトラクタを与える。ただし周期が異なる場合、全系周期は最小公倍数となり、
位相同値な直積軌道の数え方には別途規約が必要である。

## 4. ロバストレパートリの積則

外乱上限を \(e\) とし、部分系 \(k\) の全候補アトラクタ集合を
\(\mathcal A_k\)、そのうち一様外乱 \(e\) に対してcertificateを満たす集合を
\(\mathcal A_{k,\mathrm{rob}}(e)\) とする。

ブロック間に結合がなく、全系certificateが各部分系certificateの論理積と
同値なら、

\[
\mathcal A_{\mathrm{rob}}(e)
=\mathcal A_{1,\mathrm{rob}}(e)
\times\mathcal A_{2,\mathrm{rob}}(e)
\]

である。従ってロバスト個数は

\[
N_{\mathrm{rob}}(e)
=N_{1,\mathrm{rob}}(e)N_{2,\mathrm{rob}}(e)
\]

となり、全候補に対する割合

\[
R_k(e)=
\frac{N_{k,\mathrm{rob}}(e)}{|\mathcal A_k|}
\]

を用いると、

\[
\boxed{R(e)=R_1(e)R_2(e)}
\]

を得る。独立同型モジュールなら \(R(e)=R_{\mathrm{module}}(e)^2\)
である。

この積則は、単なるアトラクタ数が同じでも、モジュール数が増えるほど
全系の同時ロバスト性が急速に低下し得ることを示す。各モジュールの
ロバスト率が \(r<1\) で、\(m\) 個の独立同型モジュールを用いるなら

\[
R_m(e)=r^m
\]

である。目標全系ロバスト率 \(q\) を維持するために必要な局所率は

\[
r\ge q^{1/m}
\]

となる。これは人間規模へ外挿する際に、局所信頼度、冗長性、モジュール間
誤差訂正が不可欠となる理由を定量化する最初の下限である。

## 5. 有限時間符号記憶taskの積則

各部分系の符号と外乱cornerを一様に列挙し、全系challengeをその直積として
作る。全系の成功を「全ての部分系が評価時間内に符号を保持すること」と
定義する。このとき全系の成功は論理和ではなく論理積なので、
成功指示関数は

\[
\mathbf 1_{\mathrm{success}}
=\mathbf 1_{\mathrm{success},1}
\cdot\mathbf 1_{\mathrm{success},2}
\]

である。直積challengeを一様に平均すれば、

\[
\boxed{T(e)=T_1(e)T_2(e)}
\]

となる。\(m\) 個の独立同型モジュールでは \(T_m(e)=T_1(e)^m\)
である。

## 6. marginの直積則

アトラクタ対 \((a_1,a_2)\) の最大一様外乱marginを
\(M(a_1,a_2)\) とする。全モジュールが同じ外乱上限に耐える必要があるため、

\[
M(a_1,a_2)=\min\{M_1(a_1),M_2(a_2)\}
\]

である。したがって全系の平均marginは

\[
\bar M
=\mathbb E[\min(M_1,M_2)]
=\int_0^\infty R_1(z)R_2(z)\,dz
\]

となる。これは一般に
\(\min(\mathbb E[M_1],\mathbb E[M_2])\) とも
\(\mathbb E[M_1]\mathbb E[M_2]\) とも一致しない。

既に導入済みの恒等式

\[
\bar M=\int_0^\infty R(z)\,dz
\]

と整合し、独立モジュールでは曲線面積がロバスト率の積の積分になる。
EXP-2026-011で選択された
`normalized_mean_margin + certified_robust_fraction` は、この意味で
曲線の無次元面積と要求外乱位置での高さを同時に観測する。

## 7. スペクトルと構造診断

各2次元ブロックの固有値は符号 \(s_k\) に依存せず
\(a+g\) と \(a-g\) である。行列は実対称なのでnormalであり、

\[
\|WW^\top-W^\top W\|=0
\]

となる。4次元では非対角非零要素は4個で、ring型
`sparse_symmetric` の8個、完全結合型の12個と異なる。

このfamilyは「対称性」だけでは既知族に対して新しくない。一方で、
グラフの非連結性と力学系の厳密な積分解は既知4族に含まれない。
従ってEXP-2026-012は、単なる疎性の追加試験ではなく、積構造への外挿試験
として解釈する。

## 8. 生物学的仮説との対応

一次研究から支持されるのは次の限定された事実である。

- ヒト白質構造結合には広範で多遺伝子的、空間構造化された遺伝的影響がある。
- 発達中の機能結合は皮質階層に沿って再編され、感覚運動系と連合系で
  異なる軌道を示す。
- 多尺度の構造connectome固有モードは、安静時・課題時の機能活動を
  単一尺度の近似よりよく捉える。
- 乳幼児期の機能モジュールには、年齢に沿う共通構造と個体差の両方がある。

これらは「遺伝的に設計された最低限アトラクタと余剰アトラクタ」の直接証拠
ではない。本研究での作業仮説は、固定構造制約がロバストな基底レパートリを
形成し、可塑的自由度がその周囲に追加task能力を形成する、という計算論的
再表現である。

独立モジュール模型は、その仮説を次の検証可能な問いへ分解する。

1. 既知の非モジュール族で得たロバスト指標は、積構造へ外挿できるか。
2. モジュール数増加に伴う \(r^m\) の信頼度低下を、冗長性や弱い
   モジュール間結合で抑えられるか。
3. 固定coreの保証を維持しながら、どの結合自由度をplastic reserveとして
   解放できるか。
4. 学習後に新しいアトラクタを追加しても、coreアトラクタのmarginを
   certificate付きで保存できるか。

## 9. 未解決事項

- 弱いモジュール間結合を加えたとき、積則からの誤差を結合normで
  上下から評価できるか。
- 周期・準周期・カオスアトラクタを含むときの直積レパートリ数え上げ。
- 局所module成功率から全系task成功率を推定する際の相関補正。
- 共通外乱、局所外乱、構造外乱で積則がどのように変わるか。
- 人間規模の必要条件を、ニューロン数ではなく、有効モジュール数、
  局所ロバスト率、階層深さ、可塑的自由度、エネルギー制約で表す方法。

## 10. 参照した一次研究

- Wainberg et al., “Genetic architecture of the structural connectome,”
  *Nature Communications* 15, 1962 (2024).
  <https://doi.org/10.1038/s41467-024-46023-2>
- Sydnor et al., “Functional connectivity development along the
  sensorimotor-association axis enhances the cortical hierarchy,”
  *Nature Communications* 15 (2024).
  <https://doi.org/10.1038/s41467-024-47748-w>
- Bian et al., “Evaluating the evolution and inter-individual variability
  of infant functional module development from 0 to 5 years old” (2024).
  <https://arxiv.org/abs/2407.13118>
- Xia et al., “Multiscale structural connectome eigenmodes constrain human
  brain functional dynamics,” *Communications Biology* (2026).
  <https://doi.org/10.1038/s42003-026-10558-5>
- Raghav et al., “The Genetic and Environmental Architecture of the Human
  Functional Connectome” (2026, preprint).
  <https://arxiv.org/abs/2604.24614>
