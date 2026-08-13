# Model Card — `Churn_model` (churn propensity)

Following the business-framing / evaluation / documentation structure from the *Propensity Model* methodology (§1, §7.2): this is the artifact a reviewer reads to understand what the model does, how it was built, and where it should and shouldn't be trusted — not just its headline AUC.

---

## 1. Business framing

**Company context (this project's scenario):** a Fintech/E-commerce business facing rising CAC, churn concentrated among valuable customers, and a mass-marketing retention program that wastes budget on customers who'd stay or leave regardless, while over-treating (spamming) everyone else.

**Target action:** a customer makes **zero transactions** in the `(snapshot_date, snapshot_date + 30 days]` window. Churn = failure to transact, not account closure — there is no explicit "close account" event in this dataset, so absence of activity is the observable proxy.

**Model consumption:** the model's score is consumed as a **ranking**, not a calibrated probability. The business goal is "pick the top 20% of customers to prioritize for a retention campaign," so the metrics that matter are decile lift and capture-at-20%, not raw AUC or a perfectly calibrated probability (methodology §1.2 — "if the business only needs the top-N customers, don't blindly optimize calibration; optimize lift at the top decile"). Calibration is still measured (Brier score, calibration curve) so ROI/expected-value calculations remain possible if a future use case needs them.

**Operational constraints:**
- Scoring cadence: designed for batch scoring (weekly retrain via Airflow's `train_and_promote` DAG; ad-hoc scoring anytime via `models:/Churn_model@prod`).
- Feature latency: features are computed strictly from transactions with `transaction_date <= snapshot_date` — no same-day data assumed available at serving time beyond what's already in `data/transactions.csv`.
- No sensitive/protected attributes are used — the feature set (`RFM_v1`) is behavioral only.

**Baseline & success criterion:** the baseline is a mass-marketing campaign — contact 100% of customers, or a naive rule (e.g. "not purchased in 30 days"). Success is measured as **lift over that baseline at the top-20% cut**: does targeting only the top 20% by model score capture meaningfully more than 20% of actual churners? (See §4 — it captures ~33%, a 1.6x lift over random.)

---

## 2. Feature lineage

Feature set `RFM_v1`, defined once in [`src/feature_engineering.py`](../src/feature_engineering.py) and imported by every consumer (training, evaluation, serving) — the single-source-of-truth design already documented in the main README.

| Feature | Definition | Window |
|---|---|---|
| `recency_days` | Days since last transaction, relative to snapshot | — |
| `freq_30d` | Transaction count | trailing 30 days |
| `freq_90d` | Transaction count | trailing 90 days |
| `monetary_90d` | Total spend | trailing 90 days |
| `freq_ratio` | `freq_30d / freq_90d` — fading-behavior signal | — |

Every computation drops transactions after `snapshot_date` before aggregating (`compute_features_for_all_customers`) — the point-in-time discipline the methodology calls the "golden rule" (§2.4).

---

## 3. Training methodology

Two training modes exist side by side (`scripts/train_model.py`):

- **Single-snapshot** (default, unchanged) — one 80%-of-timeline snapshot, random 80/20 row split. This is what the Airflow `train_and_promote` DAG runs weekly via `--dataset data/processed/churn_dataset.csv`, kept as-is for pipeline stability.
- **Stacked cohorts** (`--stacked`, new) — per the methodology §2.2: a monthly snapshot every 30 days across the observable timeline (`min_history_days=90` from the start, reserving `inactivity_window=30` days at the end), each cohort computed with the same point-in-time features/labels. Rows from the same customer across cohorts are correlated, so the split is **by `customer_id`** (`GroupShuffleSplit`), never by row.

**Validation — in-time vs. out-of-time (§4.1):**
- *In-time*: a held-out 20% of customers from the training cohorts (catches ordinary overfitting).
- *Out-of-time (OOT)*: the single most-recent cohort, entirely unseen during training — the honest estimate of how the model performs on the future, which is the actual deployment condition.

**Latest run** (`--stacked`, full history through the current dataset):

| Split | ROC-AUC | Avg. Precision | Accuracy |
|---|---|---|---|
| In-time | 0.818 | 0.778 | 0.751 |
| **Out-of-time** | **0.788** | 0.733 | 0.710 |

In-time-vs-OOT gap: **0.030 AUC**. A small, expected gap — the model isn't memorizing a specific snapshot's idiosyncrasies. Per the methodology's rule of thumb (§6.1), AUC 0.75–0.85 on OOT is the "good and credible" band; anything above ~0.90 would be a leakage red flag, not a win.

Stacked-cohort dataset: 15 monthly snapshots, 33.8k total (customer, snapshot) rows, ~47% churn rate (well above the 0.5–3% the methodology describes for typical cross-sell — this dataset's 30-day inactivity definition of churn is a high base-rate label by design, not a data quality issue).

Class imbalance: `scale_pos_weight` computed from the actual train split (methodology §4.2's recommendation — LightGBM handles imbalance directly rather than SMOTE).

**Algorithm choice — LightGBM vs. XGBoost:** LightGBM was picked at project setup without ever being benchmarked against XGBoost, the other common default for tabular gradient boosting. `scripts/compare_boosting_models.py` closes that gap: both trained on the identical stacked-cohort dataset, the identical train/in-time/OOT split (same `GroupShuffleSplit` seed), and a matched hyperparameter budget (same `n_estimators`/`learning_rate`/`max_depth`/`scale_pos_weight` — a same-budget comparison, not a full tuning bake-off for either library).

| Model | In-time AUC | **OOT AUC** | Decile-1 lift | Capture@20% |
|---|---|---|---|---|
| LightGBM | 0.8182 | **0.7883** | 1.64x | 33.7% |
| XGBoost | 0.8184 | **0.7882** | 1.73x | 33.6% |

The OOT AUC difference (0.0001) is noise, not signal — on this feature set (5 numeric RFM-style features, no categoricals) neither library has a real edge. This matches expectation: LightGBM's usual advantages (native categorical handling, faster training via histogram binning + leaf-wise growth on large/wide data) don't have anything to bite into here, and both converge to the same performance ceiling. That ceiling is set by the feature set (`RFM_v1`, 5 features), not the choice of boosting library — a reason to prioritize feature work over algorithm-swapping if OOT AUC needs to move. LightGBM stays the production choice for training-speed and operational-simplicity reasons (already the established pipeline), not because it out-predicts XGBoost.

---

## 4. Evaluation — business language (methodology §6)

Run via `scripts/evaluate_business_metrics.py` against the latest scoring population:

| Metric | Value |
|---|---|
| Decile-1 lift | **1.65x** |
| Precision@20% | 80.6% |
| **Capture@20%** | **32.9%** |
| KS statistic | 0.44 |
| Brier score | 0.187 |

Read in business language: *targeting only the top 20% of customers by model score captures 32.9% of everyone who will actually churn in the next 30 days — a 1.65x lift over decile 1's baseline share, and far better than the 20% a random or mass-marketing list would capture.* This is the number behind the "top 20% priority customers" targeting decision used in `notebooks/02_propensity_targeting_campaign.ipynb`.

Decile table (decile 1 = highest predicted risk) and calibration curve are saved as artifacts by the evaluation script under `reports/business_eval/` and logged to the `Churn_model_BusinessEval` MLflow experiment. SHAP summary plot is generated in the same run (methodology §7.1) — `recency_days` and `freq_ratio` dominate, consistent with domain intuition (a customer who hasn't bought in a while, and whose recent activity is fading relative to their 90-day baseline, is the clearest churn signal). No single feature dominates by an implausible margin, which is the sanity check the methodology recommends against leakage (§6.3).

---

## 5. Monitoring

`scripts/monitor_drift.py` computes PSI (Population Stability Index, §9.1) between the training feature distribution and the current scoring population, per feature and for the model's score distribution:

- PSI < 0.10 → stable
- PSI 0.10–0.25 → watch
- PSI > 0.25 → investigate before trusting the model

This is basic, script-based monitoring — no Grafana/Prometheus dashboard for model-level drift in this iteration (infra-level monitoring already exists for the containers themselves). Intended to be run ad hoc, or wired as a scheduled Airflow task in a future iteration.

**Promotion gate** (`scripts/promote_model.py Churn_model data/transactions.csv`): candidate and the current `prod` version are both re-scored on **one identical, freshly-built holdout** — not each alias's own historically logged metric (see §6 for why that distinction matters in practice, not just in theory). Promotion requires the candidate to beat prod on fresh `roc_auc`, and to not regress on fresh `capture_at_20pct`. Without a transactions CSV argument, the script falls back to comparing logged metrics, with a warning — weaker, kept only for backward compatibility.

---

## 6. Known issue: `prod` leakage incident (found and fixed 2026-08-10)

**Symptom:** the `prod` alias (model version 1, trained 2026-06-29) reported `roc_auc = 0.981` — well above the methodology's own "AUC > 0.90 is a leakage red flag, not a win" guideline (§1, §4). Versions 2–18 (all trained 2026-07-12, before 16:14 UTC) showed the same pattern (AUC 0.92–0.95). Version 19 (trained 2026-07-12, after 16:14 UTC) and version 20 (`--stacked`, this iteration) both landed at an honest ~0.78–0.79.

**Root cause, confirmed by direct reproduction:** `compute_features_for_all_customers()` used to aggregate a customer's transactions **without first dropping rows after `snapshot_date`**. Once the dataset started growing daily (via the `simulate_data` DAG), a still-active customer (`churn=0`) would have transactions *after* the snapshot, and the buggy code picked those up as "the most recent transaction" when computing `recency_days` — leaking the label directly into the single most important feature. Reproducing both code paths on the same data and snapshot:

| | Correct (point-in-time filter) | Buggy (no filter) |
|---|---|---|
| Customers with **negative** `recency_days` (impossible) | 0 | 2,733 |
| Mean `recency_days`, churn=0 | 33.8 | **-82.9** |
| Mean `recency_days`, churn=1 | 87.8 | -4.3 |
| ROC-AUC | 0.78 | **0.95** |

This matches the version-history discontinuity almost exactly, and matches a bug the main README already documented under Engineering Highlights ("Snapshot-bounded features... caught during this project's own testing") — the fix landed in the feature-engineering code, but nobody re-ran promotion afterward.

**Why the old promotion gate could never self-correct:** `promote_model.py` compared each alias's *own* logged metric. Once `prod` held an inflated 0.98, no honestly-trained candidate could ever beat it on that basis — the leak was baked into the stored number itself, not into anything a same-metric comparison would catch. `prod` stayed pinned to a leaky version 1 for over a month of real (simulated) production time.

**Verification that the fix actually resolves it, not just explains it:** re-running the leaky version 1 through today's point-in-time-safe feature pipeline (i.e., scoring it honestly, not retraining it) gives `roc_auc = 0.745` on a fresh holdout — *below* version 20's 0.790. The apples-to-apples promotion gate (§5) was tested by deliberately resetting `prod` back to version 1 and re-running `promote_model.py`: it correctly promoted version 20 back over it automatically, with no manual intervention.

**Resolution:**
1. `prod` alias moved from v1 to v20 (manual `set_registered_model_alias` override — the only way to fix an already-stuck alias, since the old gate could never do it).
2. `promote_model.py` rewritten to score candidate and prod on one fresh, identical holdout (§5).
3. Versions 1–18 tagged `known_issue=point_in_time_leakage` in the registry (not deleted — kept for audit trail).
4. `airflow/dags/train_dag.py`'s `promote_churn_model` task updated to pass the transactions path, so the automated weekly pipeline uses the fixed comparison, not just manual CLI runs.

**Takeaway for future work:** a promotion gate that trusts each side's self-reported metric is only as trustworthy as the least-scrutinized number in the registry. The methodology's OOT/business-evaluation discipline (§3–§4) makes it easy to *build* an honest model; it does not by itself prevent an old, dishonest one from becoming permanently un-unseatable. The fix has to live in the comparison step, not just the training step.

---

## 7. Known limitations

- **High base churn rate (~47-48%)** because "churn" here means "no purchase in 30 days," not account closure — this is a much easier-to-flip label than typical subscription churn, and the business framing should account for that when comparing lift numbers to industry benchmarks.
- **No true CAC data** — this project uses revenue/cohort proxies (see `notebooks/01_eda_business_insights.ipynb`), not real acquisition spend.
- **`customer_value` is a proxy, not a trained CLV model** (annualized lifetime-average monthly spend, `compute_customer_value_proxy()` in `src/feature_engineering.py`) — deliberately a lifetime average rather than a short trailing window, since a short window collapses toward zero for exactly the customers becoming at-risk and would cancel out the churn signal in a combined priority score; still lightweight by design, and a BG/NBD + Gamma-Gamma CLV model already exists in `customer_analysis.ipynb` and could replace this proxy in a future iteration.
- **Single-product data** — the dataset has no product/category dimension, so this model (and the EDA) cannot speak to cross-sell or product-mix effects, only overall spend/frequency behavior.
- **Stacked-cohort training is opt-in** (`--stacked` flag) — the Airflow production DAG still trains on the single-snapshot dataset for pipeline stability; promoting stacked-cohort training to the default path is a follow-up decision, not made in this iteration.
