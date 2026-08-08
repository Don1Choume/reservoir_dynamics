# 空間変調場による機能core保護とplastic reserveの選択的開放

最終確認: 2026-08-02  
状態: 生物学的着想を抽象化した理論候補。場の不変性とcore保護条件は解析的に
導出するが、生物学的同定と数値的な設計優位は未確認。

## 1. 証拠境界

### 一次研究から支持される事実

- 2024年のmouse cortex研究では、局所かつ短時間のGABAまたはglutamate入力が、
  単一astrocyteを越えて広がり、分単位で持続するCa2+ network応答を生じた。
  [Cahill et al. 2024](https://doi.org/10.1038/s41586-024-07311-5)
- 2024年のmouse hippocampus研究では、astrocyteの末梢過程からsomaへ向かう
  秒単位のCa2+伝播が観測され、locus coeruleus刺激とalpha1 adrenergic receptor
  操作がこの統合へ関与した。
  [Centripetal integration study 2024](https://doi.org/10.1038/s41593-024-01612-8)
- 2024年のmouse striatum研究では、astrocyteの一過性脱分極が局所の細胞外
  Ca2+を変え、cholinergic interneuronの発火とdopamine放出をsubsecondで
  変調した。ただし興奮・抑制の向きは細胞群で一様ではなかった。
  [Rapid astrocyte modulation study 2024](https://doi.org/10.1038/s41467-024-54253-7)
- 2026年のmouse brain研究では、astrocyteのNa+濃度とNa+/K+-ATPaseを介した
  extracellular K+取込み能力に、細胞間・細胞内subdomain間の不均一性があり、
  周囲回路に応じた局所homeostasisと整合する結果が得られた。
  [Astrocytic Na+ homeostasis study 2026](https://doi.org/10.1038/s41467-026-73435-z)
- striatumのACh–dopamine相互作用には局所放出と数十micrometreの影響が観測され、
  activator–inhibitor reaction–diffusion modelが波動を再現した。
  [ACh–dopamine waves 2023](https://doi.org/10.1038/s41467-023-42311-5)
- dopamine依存plasticityと一過性excitability変調を分けたmouse行動・回路model
  では、plasticityが形成した潜在的attractorをexcitability変調が広いbasinから
  利用可能にするという予測が行動介入と対応した。
  [Dopamine latent attractors 2024](https://doi.org/10.1038/s41467-024-53976-x)

### 本研究で導入する推論

上記は、astrocyte、ion、neuromodulatorが本研究のcore–reserve分割を実装して
いることを直接示さない。本研究では共通する最小構造だけを抽出する。

1. 神経状態より遅い内部状態を持つ。
2. 局所入力が空間的に伝播する。
3. 場の作用はnode、edge、受容体分布により不均一である。
4. 同じ場が短期excitabilityと長期plasticityを別経路で変え得る。

この抽象化は、特定のhormone、ion、astrocyte Ca2+を同定した生物物理modelでは
ない。現段階では、複数の異なる機構が同じ数理形式へ射影され得る。

## 2. 離散空間場

nodeまたは局所領域上の抑制場を \(z_t\in[0,1]^n\)、非負row-stochasticな
拡散kernelを \(P\)、局所sourceを \(s_t\in[0,1]^n\) とし、

\[
z_{t+1}
=
(1-\alpha-\beta)z_t
+\alpha Pz_t
+\beta s_t
\]

とする。\(\alpha\ge0\)、\(\beta\ge0\)、\(\alpha+\beta\le1\) とする。

### 命題1: 場のhypercube不変性

\(z_0\in[0,1]^n\)、\(s_t\in[0,1]^n\) なら、すべての \(t\) で
\(z_t\in[0,1]^n\) である。

#### 証明

row-stochastic性より \(Pz_t\in[0,1]^n\) である。更新式は
\(z_t\)、\(Pz_t\)、\(s_t\) と0の非負係数による凸結合なので、各座標は
\([0,1]\) に残る。帰納法で従う。

この式はreaction–diffusion、volume transmission、glial networkを一意に
表すものではない。まず数値検証で追跡可能な最小field adapterとして用いる。

## 3. core–reserveへの局所作用

core \(c_t\)、reserve \(r_t\) を

\[
c_{t+1}
=
\tanh\left(
W_c c_t + D(g_t)G r_t + B_cu_t+\eta^c_t
\right),
\]

\[
r_{t+1}
=
\tanh\left(
W_r r_t + Hc_t + B_ru_t+\eta^r_t
\right)
\]

とする。\(G\) はreserveからcoreへのfeedback、\(H\) はcoreからreserveへの
結合であり、一般に非対称かつmodule sizeは異なってよい。

抑制場をfeedback targetごとのgate

\[
g_{i,t}=1-z_{i,t}
\]

へ写す。局所gateは \(D(g_t)G\) だけを変え、\(W_c\) と \(W_r\) を保存する。
これは「局所ion変化が必ずfeedback edgeだけを変える」という生物学的主張では
なく、core保護に必要な介入scopeを同定するための工学的比較である。

## 4. 保護certificate

### 4.1 収縮core

\(L_c=\|W_c\|_\infty<1\) とし、同じ入力・core外乱を受けるzero-feedback
参照軌道との距離を \(D_t=\|c_t-c_t^0\|_\infty\) とする。tanhの
1-Lipschitz性から

\[
D_{t+1}
\le
L_cD_t+\ell_t,
\qquad
\ell_t=\|D(g_t)Gr_t\|_\infty.
\]

従って

\[
D_t
\le
L_c^tD_0
+
\sum_{k=0}^{t-1}L_c^{t-1-k}\ell_k.
\]

定数norm上界だけでなく、実際の場・reserve状態・edge方向を用いた時変負荷を
保存すれば、`EXP-2026-005` の一様上界より条件依存のtightな保証を作れる。

### 4.2 対角双安定core

各core座標が

\[
c_{i,t+1}=\tanh(a_ic_{i,t}+\xi_{i,t}),\qquad a_i>1
\]

であり、

\[
m_i=\sqrt{1-1/a_i},
\qquad
\mu_i=a_im_i-\operatorname{atanh}(m_i)
\]

とする。局所gate後のfeedbackと外乱が

\[
|(D(g_t)Gr_t)_i+\eta^c_{i,t}|\le\mu_i
\]

を全時刻で満たせば、初期符号に対応する区間
\([m_i,1]\) または \([-1,-m_i]\) はロバスト正不変である。従って
\(d_c\) 座標の直積で表される \(2^{d_c}\) 個の必須orthantを同時に保護できる。

これは対角coreに対する十分条件であり、非対角・非normal coreの必要条件では
ない。

## 5. intervention energyを一致させたglobal対照

局所gateによる時刻 \(t\) の重み摂動energyを

\[
E_t^{local}=\|D(g_t)G-G\|_F^2
\]

とする。全recurrent matrixを一様に \((1-q_t)W\) へ縮小するglobal対照は、

\[
q_t
=
\min\left(
1,
\sqrt{E_t^{local}/\|W\|_F^2}
\right)
\]

とすれば \(\|q_tW\|_F^2=E_t^{local}\) を満たす。この対照により、局所gateの
優位が単に大きな介入energyによるものか、task-relevant edgeへ集中したことに
よるものかを分離する。

予測は、同じenergyなら局所gateがcore orthant保持とreserve記憶の双方で
global対照をPareto改善することである。これは `EXP-2026-015` で反証する。

## 6. 未解決部分

- sourceをfeedback loadから作る規則はhomeostatic sensorの抽象であり、特定の
  ion sensorまたはreceptor kineticsへ同定されていない。
- hormone、monoamine、astrocyte fieldは時間・空間scaleが異なる。単一の
  \(\alpha,\beta\) で同一視してはならない。
- 場がplasticity ruleそのものを変える二重時間scaleは未実装である。
- global対照は一様gain制御の一例であり、最適な低rank制御やMPCではない。
- 高次元でfield node数を増やしたときのcommunication、energy、遅延のscale lawは
  未導出である。
