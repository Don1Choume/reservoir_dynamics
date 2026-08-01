# 符号座標共役と有効構造標本数

更新日: 2026-08-01

## 1. 目的

network generatorのseed数は、そのまま異なる力学系の標本数とは限らない。
特に成分ごとのtanhを持つRNNでは、座標の符号を変えただけの重みは力学的に
共役である。本稿は、その同値性、検出算法、taskまで同値とみなせる条件、
既存5 familyの監査結果を固定する。

## 2. 力学系の共役

離散時間RNN

\[
x_{t+1}=\tanh(Wx_t+u_t)
\]

と、対角成分が \(p_i\in\{-1,+1\}\) の符号行列
\(D=\operatorname{diag}(p_1,\ldots,p_d)\) を考える。成分ごとのtanhは奇関数
なので、\(\tanh(Dz)=D\tanh(z)\) である。座標を \(y=Dx\) と置けば、

\[
y_{t+1}=\tanh(DWD\,y_t+Du_t)
\]

を得る。従って

\[
W'=DWD
\]

なら、入力も同時に \(u'_t=Du_t\) へ写した二系は共役である。固定点、周期、
Lyapunov spectrum、吸引域体積など、座標符号に依存しない量は保存される。

## 3. 辺制約と閉路不変量

非零要素ごとに

\[
W'_{ij}=p_iW_{ij}p_j
\]

が必要である。絶対値と零patternが一致するとき、これは

\[
p_ip_j=\operatorname{sign}(W'_{ij}/W_{ij})
\]

という二値制約になる。あるnodeの符号を+1に固定し、隣接辺に沿って符号を
伝播すれば証人候補を構成できる。閉路を一周して矛盾する場合は共役でない。
従って、絶対値行列だけを比較する方法と異なり、cycle sign productを保存する。

連結成分ごとのglobal signは作用を変えないため、各成分の最小indexを+1に
固定すれば証人を決定論的に一つ選べる。実装は
`signed_coordinate_conjugacy_witness` と
`audit_signed_coordinate_conjugacy` に置いた。

## 4. 理論上のclass数

独立に符号化されたedge parameterが \(E\) 個あり、そのsupport graphが
\(V\) node、\(C\) connected componentを持つとする。各parameterが
\(p_ip_j\) の作用を受け、全符号配置が許される場合、作用rankは \(V-C\) で、
符号共役class数は

\[
2^{E-(V-C)}=2^{E-V+C}
\]

である。これは独立parameterの数え方に依存する。対称な往復辺が同じ符号を
共有する場合は一parameter、独立な有向辺なら二parameterとして数える。

4 nodeの既存generatorでは次を得る。

| family | 独立符号parameter \(E\) | \(V-C\) | 理論class数 |
|---|---:|---:|---:|
| dense symmetric K4 | 6 | 3 | 8 |
| sparse symmetric C4 | 4 | 3 | 2 |
| feedforward K4 DAG | 6 | 3 | 8 |
| modular two-edge forest | 2 | 2 | 1 |
| asymmetric dense digraph | 12 | 3 | 512 |

## 5. task同値性の追加条件

重みが共役でも、固定された一方向入力、特定nodeだけのreadout、非対称な初期値
集合ではtask性能が同じとは限らない。taskまで同値とみなせるのは、少なくとも
次が符号変換で閉じている場合である。

1. 初期状態集合
2. 入力・外乱系列集合とその確率重み
3. readoutまたは成功判定
4. 集約時の標本重み

EXP-2026-012は全orthant初期値と全corner一定外乱を一様列挙し、元orthantの
符号保持を成功とした。この集合は任意の符号行列で単に置換されるため、
`modular_paired` のseed別集約値は厳密に同じになる。

## 6. AUDIT-2026-001

EXP-2026-011の確認seed 1201–1230とEXP-2026-012のseed 1301–1330を、
family・gainごとに監査した。

| family | raw / gain | effective / gain | raw / effective |
|---|---:|---:|---:|
| dense symmetric | 30 | 8 | 3.75 |
| sparse symmetric | 30 | 2 | 15.00 |
| asymmetric dense | 30 | 29 | 1.03 |
| feedforward non-normal | 30 | 8 | 3.75 |
| modular paired | 30 | 1 | 30.00 |

全20 family-gain群ではraw 600 networkが192 classへ縮約された。観測class数は
上表の理論上限と整合し、asymmetric denseだけは512候補中29 classを30 seedで
観測した。詳細は
`docs/research/artifacts/AUDIT-2026-001-structure-summary.json` に固定した。

## 7. 統計設計への含意

- seed bootstrapは、taskを保存する共役class内の複製を独立標本として
  resampleしてはならない。
- 推定対象が構造母集団なら、classをcluster単位として分割・bootstrapする。
- generatorの実行前gateとしてraw seed数、有効class数、最大class size、
  class entropyを報告する。
- 連続重みでもpermutation、scale、一般のsimilarity、task固有対称性が残り得る。
  符号対角共役監査だけで完全な独立性は保証しない。

## 8. 未対応範囲

現在のAPIは座標符号だけを扱い、node permutationを同時に含むsigned
permutation、一般の線形・非線形共役、近似共役は未実装である。また、task
閉性を自動判定せず、利用者が実験specから別途監査する。EXP-2026-013では
module coupling magnitudeをseedごとに変え、少なくとも符号共役による崩壊を
除いたうえで積則を検証する。

