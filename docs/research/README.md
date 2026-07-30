# 研究証拠の運用規約

このディレクトリは、理論、既存研究、実験、実装の対応関係を追跡する。
目的は、興味深い説明を増やすことではなく、どの主張がどの証拠で支えられ、
何が未解決かを第三者が再検査できる状態に保つことである。

## 証拠の四分類

| `evidence_type` | 意味 | 記述上の規則 |
|---|---|---|
| `sourced_fact` | 論文や公式資料が直接支持する事実 | 出典が述べた範囲を越えない |
| `local_reproduction` | 本リポジトリで事前条件付き再現に成功した結果 | spec、seed、artifact、失敗率を添える |
| `inference` | 複数の事実や数学的定義から導いた推論 | 前提と導出を明示する |
| `hypothesis` | 現時点で未検証の予測 | 反証条件と次の実験を必須にする |

主張の検証状態は別軸で管理する。

| `status` | 意味 |
|---|---|
| `established` | 査読済み資料または数学的定義で裏付けられた範囲 |
| `provisional` | プレプリント、単一研究、適用範囲未確認の主張 |
| `reproduced` | ローカルの事前規定した条件で再現済み |
| `hypothesis` | 反証可能だが未検証 |
| `refuted` | 規定条件で反証された主張 |

`established` は「本研究のあらゆる条件で正しい」を意味しない。
出典の仮定と適用限界を `limitations` に残す。

## 一つの研究サイクル

1. `claims.toml` に主張ID、状態、出典、限界、反証条件を登録する。
2. `theory/` で記号、仮定、導出、数値推定量を分離して記述する。
3. `experiments/TEMPLATE.md` から実験記録を作り、評価前に判定条件を固定する。
4. 実装は主張IDをdocstringまたは設計文書から参照し、テストを先に書く。
5. raw artifactを保存し、集計値だけで結論を更新しない。
6. 成功・失敗の双方を台帳へ反映し、都合の悪いseedを除外しない。
7. 文献更新時は `literature-watch.md` の確認日と差分を更新する。

## 状態遷移

```text
hypothesis
  ├─ 追試で支持 ─> reproduced
  ├─ 外部資料のみ ─> provisional
  └─ 反証 ──────> refuted

established
  └─ 適用条件外では新しいhypothesisとして分岐
```

ローカル再現だけで一般的な `established` へ昇格させない。異なる力学系family、
未知seed、未知入力分布への外的妥当性を別々に検証する。

## ファイル

- `claims.toml`: 機械可読な主張台帳
- `open-questions.md`: 未解決課題と次の識別実験
- `literature-watch.md`: 継続調査の検索軸と最終確認日
- `theory/core-metrics.md`: 実装済み基礎指標の数学的定義
- `theory/core-reserve-protection.md`: 機能coreを保つreserve更新の十分条件
- `theory/bistable-core-margin.md`: 多重安定coreのロバスト不変集合と外力margin
- `theory/orthant-box-margin.md`: 非対角RNNのorthant別robust repertoire
- `experiments/EXP-2026-010.md`: 4 network familyの未知seed task確認
- `experiments/TEMPLATE.md`: 実験記録テンプレート
- `directions/innate-core-plastic-reserve.md`: 生得的構造と生涯学習の長期仮説
- `../papers/robust-repertoire-memory-ja.md`: robust repertoire論文草稿

台帳は `load_claim_registry` により検証され、仮説に反証条件がない場合や、
検証済み扱いの主張に出典がない場合はテストが失敗する。
