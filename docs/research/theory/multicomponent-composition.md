# 未知分割を含む多成分合成certificate

更新日: 2026-08-02  
対応仮説: `H-RC-008`, `H-BIO-004`

## 1. 目的

二module用の方向別certificateを、未知の分割を持つ (m\ge3) 個のmoduleへ
拡張する。保証、構造推定、経験的task predictorを分離し、人間規模へ外挿する
ために必要な計算量条件と負荷条件を陽にする。

## 2. 多成分系と方向別負荷行列

状態座標を互いに素な集合 (I_1,\ldots,I_m) へ分割し、

\[
x_i^+=\tanh\!\left(W_{ii}x_i+
\sum_{j\ne i}W_{ij}x_j+\eta_i\right)
\]

とする。孤立module (i) のorthant (s_i) に対するrectangle marginを
(M_i(s_i))、外乱budgetを (e) とする。module (j) から (i) への
最大流入量を

\[
L_{ij}=\max_{r\in I_i}\sum_{c\in I_j}|W_{rc}|,
\qquad
\ell_i=\sum_{j\ne i}L_{ij}
\]

と定義する。tanh状態は成分ごとに ([-1,1]) にあるため、任意時刻のcross入力は
(\ell_i) 以下である。従って

\[
\boxed{M_i(s_i)\ge e+\ell_i\quad(i=1,\ldots,m)}
\]

なら、孤立rectangleの直積は成分別外乱 (e) と全cross入力に対してロバスト
正不変である。これは有限rectangle版の方向別small-gain条件であり、必要条件
ではない。

## 3. 因子化できる認証率

module (i) の局所orthant集合上で

\[
r_i(a)=\frac{1}{2^{d_i}}
\sum_{s_i}\mathbf{1}\{M_i(s_i)\ge a\}
\]

と置く。方向別norm条件はmoduleごとに分離するため、全系orthantを列挙せず

\[
R_{\mathrm{dir}}(e)
=\prod_{i=1}^m r_i(e+\ell_i)
\]

を厳密に計算できる。単一global load
(\ell_{\max}=\max_i\ell_i) を全moduleへ課す場合は

\[
R_{\mathrm{global}}(e)
=\prod_{i=1}^m r_i(e+\ell_{\max})
\le R_{\mathrm{dir}}(e)
\]

である。実際のcross-edge符号と座標別境界を用いるtransported rectangleの
認証率を (R_{\mathrm{rect}}) とし、結合後固定点が対応rectangle内にある
有限検証領域では

\[
T(e)\ge R_{\mathrm{rect}}(e)
\ge R_{\mathrm{dir}}(e)
\ge R_{\mathrm{global}}(e)
\]

を監査できる。最初の二量は検証用には全orthantを列挙するが、規模外挿に
必要な安全側下界は (R_{\mathrm{dir}}) だけで計算できる。

## 4. 最小slack分布の積を列挙しない計算

局所orthantを一様に選んだときのslackを

\[
S_i=M_i-e-\ell_i,\qquad Z=\min_iS_i
\]

とする。全直積を作らなくても、任意の実数 (z) に対し

\[
\Pr[Z\ge z]=\prod_i\Pr[S_i\ge z]
\]

である。全局所slackの有限supportを昇順に走査すれば

\[
\mathbb{E}[Z]=\sum_z z\Pr[Z=z]
\]

を厳密に得られる。実装ではこのsurvival積を用い、直積配列を生成しない。

## 5. 分割回復の十分条件

観測重みから対称affinityを

\[
A_{uv}=\max(|W_{uv}|,|W_{vu}|)
\]

とする。真の各module内affinityが

\[
\alpha=\min_{u,v\text{ in same module}}A_{uv}
\]

以上、module間affinityが

\[
\beta=\max_{u,v\text{ in different modules}}A_{uv}
\]

以下で、

\[
\boxed{\alpha>\beta}
\]

かつ各module内のthreshold graphが連結なら、任意の
(\tau\in(\beta,\alpha)) における連結成分は真の分割と一致する。本研究の
自動推定器はaffinityの隣接値間で最大のgapを選ぶ。この方法が上のgapを選ぶ
保証には、inter/intra gapが他のaffinity gapより一意に大きいという追加仮定が
必要である。一般の脳・RNNでこの分離を仮定せず、回復confidenceと感度を必ず
報告する。

### 5.1 最大gap推定器のentrywise摂動保証

全pair affinityを重複を残して昇順に並べ、その隣接gapの最大値を \(g_1\)、
二番目を \(g_2\) とする。最大gapが一意なとき

\[
r_{\mathrm{edge}}=\frac{g_1}{2},\qquad
r_{\mathrm{select}}=\frac{g_1-g_2}{4},
\]

\[
\boxed{r_{\mathrm{part}}=
\min(r_{\mathrm{edge}},r_{\mathrm{select}})}
\]

と定義する。重み摂動を \(\Delta W\) とすると、

\[
\left|\max(|W_{uv}+\Delta W_{uv}|,
|W_{vu}+\Delta W_{vu}|)-A_{uv}\right|
\le\|\Delta W\|_{\max}
\]

である。従って \(\|\Delta W\|_{\max}<r_{\mathrm{part}}\) なら、選択gapは
最大 \(2\|\Delta W\|_{\max}\) しか縮まず、任意の競合gapも最大同量しか
拡大しない。gap順位とthreshold前後のedge集合が保存されるため、連結成分も
不変である。等号ではtieが生じ得るので保証しない。

この半径は必要条件ではない。\(r_{\mathrm{part}}\) を越えてもpartitionが保たれる
場合があり、半径が0でも別の推定法なら安定に回復できる可能性がある。従って
分析器は一点partitionだけでなく、\(g_1\)、\(g_2\)、gap dominance、絶対・相対
保証半径を返す必要がある。

### 5.2 分割差のlabel-free測定

二partition \(P,Q\) について、unordered node pairの同一component判定が異なる
割合

\[
d_{\mathrm{pair}}(P,Q)=\binom n2^{-1}
\sum_{u<v}\mathbf 1\{[u\sim_Pv]\ne[u\sim_Qv]\}
\]

を用いる。これはmodule labelに依存せず、0なら二partitionは同値である。
ただし一点間の距離だけでは複数の競合consensusを要約できないため、摂動半径外では
partition頻度、node-pair共所属確率、複数modeを保存する必要がある。

## 6. 人間規模へ向けた計算量条件

全状態次元を (n=\sum_i d_i)、最大局所次元を (b=\max_i d_i) とする。
monolithicなorthant列挙は (2^n) だが、局所profileと方向別合成は

\[
\sum_i2^{d_i}\le m2^b
\]

である。従って次が同時に必要になる。

1. (b) が全体規模に対して有界または緩やかにしか増えない。
2. 分割が構造または活動から再現可能で、推定誤差を監査できる。
3. coupling graphが疎で、(L_{ij}) の計算と更新が局所edge数に比例する。
4. 安全余裕が消えないよう
   \(\sup_i\ell_i<\inf_{i,s_i}M_i(s_i)-e\) を規模に依存せず保つ。
5. module数 (m) が増えても全系成功率 (q) を保つなら、独立近似下で各局所
   成功率は (p_i\gtrsim q^{1/m}=1+\log(q)/m+O(m^{-2})) を満たす。

特に4は、次数が増える場合にcross weightを次数で正規化する、局所marginを
増やす、または局所gateで有効流入を抑える必要があることを示す。これは
「module化だけで人間規模になる」という主張を否定し、規模とともに維持すべき
設計不変量を与える。

## 7. 既存研究と生物学的対応

- ISS Lyapunov関数をsubsystemごとに構成して大規模networkへ合成する研究は、
  局所certificate再利用と規模外挿の先行例である。本研究は学習Lyapunov関数
  ではなく、tanh orthant rectangleと有限外乱taskを扱う。
  <https://proceedings.mlr.press/v211/zhang23a.html>
- 2025年のsISS preprintは離散時間の相互結合系で規模に依存しないcertificateを
  扱う。本研究に直接適用済みではなく、査読前の比較対象である。
  <https://arxiv.org/abs/2509.10118>
- 2026年のhuman connectome解析は、発生時期と構造中心性、近い神経発生時期を
  持つ領域間の接続、発生関連遺伝子発現の関連を報告した。発生blueprintと
  module priorの着想を支持するが、因果的なアトラクタ分割を示さない。
  <https://doi.org/10.1038/s41467-025-67785-3>
- 2025年のhuman cortex atlas統合は、発生中の細胞subtype指定に対応する
  時空間的gene co-expression meta-moduleを同定した。これは細胞型生成規則の
  生物学的根拠であり、力学moduleや安全marginの証拠ではない。
  <https://doi.org/10.1038/s41593-025-01933-2>
- network communityの小摂動に対するrobustnessは、構造の統計的有意性を測る
  既存の考え方である。本節は一般のcommunity分布ではなく、最大gap推定器に
  限定した決定論的半径を与える。
  <https://doi.org/10.1103/PhysRevE.77.046119>
- competing partitionが多峰的な場合に単一consensusが不十分であることは、
  半径外でpartition ensembleを保持すべき根拠になる。
  <https://doi.org/10.1103/PhysRevX.11.021003>
- 2025年の人工RNN研究では、構造module性だけでは機能specializationを保証せず、
  環境分離、資源制約、情報流の時空間条件が必要だった。従って分割保証半径を
  機能保証と同一視せず、task指標を別phaseで検証する。
  <https://doi.org/10.1038/s41467-024-55188-9>
- 2025年のhuman connectome研究では機能結合の個体差がwithin-networkから
  between-networkへ連続的に変わり、構造結合変動とも関連した。これは個体差を
  一様noiseではなく構造化摂動として扱う必要性を支持するが、本半径の実測検証
  ではない。<https://doi.org/10.1073/pnas.2420228122>

## 8. 証拠境界

分割回復定理は明示したaffinity gapを持つnetworkに限る。因子化certificateは
安全側の十分条件であり、認証不能は機能不能を意味しない。局所orthant数も
module次元には指数的であり、連続・周期・カオスattractor、spiking network、
学習中に変化する分割には追加の局所表現が必要である。人間脳が本条件を満たす
こと、発生時期moduleが計算moduleと一致することは未検証である。
