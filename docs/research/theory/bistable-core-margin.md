# 双安定coreのロバスト不変margin

最終確認: 2026-07-30  
対応主張: `C-DYN-003`, `C-RC-015`, `C-RC-016`, `H-BIO-005`

## 1. 目的

大域収縮しない多重安定coreに新しいreserveアトラクタを追加するとき、既存の
coreアトラクタを壊さないfeedback条件を定式化する。

本節はscalar tanh系で閉形式にできるground-truth certificateである。一般の
高次元RNN、脳、生物発生に同じ式がそのまま成立するとは主張しない。

## 2. basin stability、survivability、不変性

三つの量を区別する。

- basin stabilityは指定した初期条件分布から対象アトラクタへ漸近到達する
  確率である。
- survivabilityは軌道が望ましい領域から一度も出ない初期条件の割合であり、
  過渡応答を含む。
- ロバスト正不変集合は、許容した任意の外乱系列に対して軌道が集合内へ
  留まり続ける決定論的certificateである。

対象アトラクタのbasin内にいても、外乱後に安全領域から出ないとは限らない。
本研究では漸近的なアトラクタ所属だけでなく、学習中・学習後の機能保持を
扱うため、後二者を測る。

- Menck et al. (2013): <https://doi.org/10.1038/nphys2516>
- Hellmann et al. (2016): <https://doi.org/10.1038/srep29654>
- Jiang and Wang (2001):
  <https://doi.org/10.1016/S0005-1098(01)00028-0>

## 3. scalar tanh双安定core

\[
c_{t+1}=\tanh(ac_t+\eta_t),
\qquad a>1,
\qquad |\eta_t|\le\bar\eta
\]

を考える。無外力では原点が不安定固定点、正負に一つずつ安定固定点を持つ。

### 命題1: ロバスト正不変区間

\(m\in(0,1)\) とする。

\[
\bar\eta\le am-\operatorname{atanh}(m)
\]

なら、\([m,1]\) と \([-1,-m]\) は全ての許容外力系列に対してロバスト正不変
である。

### 証明

tanhは単調増加である。任意の \(c_t\in[m,1]\) と
\(\eta_t\in[-\bar\eta,\bar\eta]\) に対し、

\[
c_{t+1}
\ge
\tanh(am-\bar\eta)
\ge
\tanh(\operatorname{atanh}(m))
=m.
\]

tanhの値域から \(c_{t+1}\le1\) でもある。負側は写像の奇対称性から従う。
\(\square\)

### 命題2: 最大対称forcing margin

命題1で認証できる \(\bar\eta\) の最大値は、

\[
m_*=\sqrt{1-\frac1a},
\]

\[
\eta_{\mathrm{crit}}(a)
=
am_*-\operatorname{atanh}(m_*)
\]

である。

### 証明

\[
q(m)=am-\operatorname{atanh}(m)
\]

と置くと、

\[
q'(m)=a-\frac1{1-m^2}.
\]

\(q'(m)=0\) の唯一の内点解が \(m_*\) であり、

\[
q''(m)=-\frac{2m}{(1-m^2)^2}<0
\]

なので最大点である。\(\square\)

接点では

\[
a(1-m_*^2)=1
\]

が成り立つ。一定最悪外力
\(\eta_t=-\bar\eta\) に対する固定点条件を

\[
\bar\eta=ac-\operatorname{atanh}(c)
\]

と書けば、同じ接点で正の安定固定点と不安定固定点が衝突する。従って
\(\eta_{\mathrm{crit}}\) はsaddle-node tippingの臨界外力でもある。

## 4. 認証割合と吸引域の違い

無外力の正負アトラクタのbasinは、原点を除くそれぞれの半区間である。一方、
ロバストcertificateが覆うのは

\[
[-1,-m_*]\cup[m_*,1]
\]

だけである。初期条件が \(U[-1,1]\) なら、認証される割合は

\[
P_{\mathrm{cert}}=1-m_*.
\]

これはbasin stabilityではなく、指定外力集合に対するsurvivabilityの保守的
下界である。basin境界に近い状態は無外力では正しいアトラクタへ到達しても、
臨界値未満の外力で符号を失い得る。

\(a=1.5\) では、

\[
m_*=0.5773502692,
\quad
\eta_{\mathrm{crit}}=0.2075464553,
\quad
P_{\mathrm{cert}}=0.4226497308.
\]

## 5. cue形成reserveとの合成

reserveを

\[
r_{t+1}=\tanh(gr_t+bu_t),
\qquad g>1
\]

とする。最初の一時刻だけ符号付きcueを与え、その後 \(u_t=0\) とすると、
reserveはcue符号に対応する正負の自律固定点へ到達できる。

coreが

\[
c_{t+1}=\tanh(ac_t+\gamma r_t)
\]

でreserveからfeedbackを受け、\(|r_t|\le1\) なら、

\[
|\gamma|\le\eta_{\mathrm{crit}}(a)
\]

は二つのcore認証区間を任意のreserve軌道から保護する十分条件である。

この条件はreserveの実際の到達振幅、符号相殺、gatingを使わない。そのため
安全側に保守的だが、学習内容に依存しないworst-case certificateになる。

## 6. 多module・高次元への一般化

一般のcore写像を

\[
\boldsymbol c_{t+1}
=
f_c(\boldsymbol c_t,\boldsymbol\eta_t)
\]

とし、必須機能 \(k\) の安全集合を \(S_k\)、許容外力集合を
\(\mathcal U_k\) とする。

\[
\boldsymbol c\in S_k,\quad
\boldsymbol\eta\in\mathcal U_k
\Longrightarrow
f_c(\boldsymbol c,\boldsymbol\eta)\in S_k
\]

を示せれば、\(S_k\) はロバスト正不変である。

外力が \(J\) 個のreserveから

\[
\boldsymbol\eta_t
=
\sum_{j=1}^{J}G_j\boldsymbol r_{j,t}
\]

として流入し、選んだnormで
\(\|\boldsymbol r_{j,t}\|_j\le R_j\) とする。安全集合 \(S_k\) が
\(\|\boldsymbol\eta\|_*\le\mu_k\) に対してロバストなら、

\[
\sum_{j=1}^{J}
\|G_j\|_{*\leftarrow j}R_j
\le
\mu_k
\]

は機能 \(k\) を保つ十分条件である。全必須機能を同時保護するには、

\[
\sum_{j=1}^{J}
\|G_j\|_{*\leftarrow j}R_j
\le
\min_k\mu_k
\]

で足りる。

これは多数moduleへ拡張可能な設計形だが、人間規模の必要条件ではない。
高次元では \(S_k\)、\(\mu_k\)、入力方向、非normal増幅をどう認証するかが
未解決である。

## 7. EXP-2026-006による照合

[EXP-2026-006](../experiments/EXP-2026-006.md) では、

- \(a=1.2,1.5,2.0\) の全てで臨界比未満の一定最悪外力が認証区間を保持した。
- 臨界比超過では全条件が正のbranchを失い、負へtippingした。
- cue形成reserveを用いた30 seed評価で、外力比0.5と0.9のcertified
  retentionは1だった。
- 同じ条件でも無外力basin全体からの保持率は0.9341と0.8693であり、
  basin所属と外乱下survivabilityを分離した。
- 外力比1.1と1.5では、反対向きreserve cueの保持率が0になった。

## 8. 適用限界

- scalar、決定論的、離散時間tanh写像に限定した閉形式結果である。
- reserve adaptationは候補parameter選択であり、gradient学習ではない。
- noise、時変core入力、有限精度境界、非対称アトラクタを扱っていない。
- \(m_*\) は最大forcing marginを与える境界であり、最大ロバスト不変集合
  全体を一般に与えるとは主張しない。
- 外力比1以上は必ず失敗するという必要条件ではない。外力の方向・時系列が
  有利なら保持できる。
- 生物学的なmodule、発達、遺伝的指定を検証していない。

## 9. 高次元化の実装候補

2025年のpreprintでは、neural-network dynamical systemに対し、状態空間を
hyperboxへ量子化し、one-step returnable subsetの集合再帰からcontrol
invariant setを有限回で構成する方法が提案されている。

- Li et al. (2025): <https://arxiv.org/abs/2505.11546>

これはreservoirの学習余剰を扱う方法ではなく、2026-07-30時点で査読前で
ある。ただし次の実装段階で、

1. 小次元RNNの安全集合をhyperbox近似する
2. reserve出力をbounded disturbanceとして集合伝播する
3. Monte Carlo survivabilityと認証集合の被覆率を比較する

ための有力なbaselineになる。高次元ではhyperbox数が急増するため、局所断面、
zonotope、interval bound propagation、sampling certificateとの計算量比較が
必要である。
