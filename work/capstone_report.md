# Capstone Report — Content Refresh & Search Decay Prioritization

- **Author:** Jainesh Chaurasiya
- **Email:** aw888117@gmail.com
- **Lane:** Content Refresh & Search Decay Prioritization
- **Repo:** [https://github.com/jaineshchaurasiya20/FlyRank_Ml_Assignment](https://github.com/jaineshchaurasiya20/FlyRank_Ml_Assignment)
- **Deployed Paper:** [https://jaineshchaurasiya20.github.io/FlyRank_Ml_Assignment/](https://jaineshchaurasiya20.github.io/FlyRank_Ml_Assignment/)
- **Date:** September 2026

---

## 0. Abstract

Modern search engines continuously recalibrate organic rankings, causing mature website content to experience silent traffic decay that editorial teams typically detect months too late. Working with the FlyRank AI Search Intelligence dataset spanning 79M+ warehouse event rows and a 30,000-page cross-client benchmark across 32 enterprise domains, we evaluate whether machine learning can prioritize recoverable content decay more effectively than heuristic rules. We trained and audited a Gradient Boosted Decision Tree (GBDT) pipeline under a strict client-holdout validation protocol, explicitly eliminating label leakage and domain-identity memorization. On unseen client domains with a 39.1% decay base rate, the model achieves a Precision@50 of 86.0% and a PR-AUC of 0.6772, significantly outperforming the production heuristic rule baseline (Precision@50 = 22.0%, PR-AUC = 0.4700). We operationalize these predictive rankings into a human-in-the-loop Content Action Playbook that maps multi-dimensional search signals into concrete editorial workflows with clear cost-benefit economics, safety guardrails, and retrain tripwires.

---

## 1. Problem Framing

### Decision Context & Operational Unit
Content marketing and organic SEO represent substantial corporate investments, yet search rankings decay dynamically as algorithms update, competitors publish fresher content, and user query preferences evolve. Editorial teams face an acute triage challenge: out of thousands of published articles in an enterprise content catalog, which specific pages should human writers and SEO specialists prioritize for updates each sprint?

- **Unit of Analysis:** One published URL (`content_id`) within a client catalog (`client_id`), evaluated over a 90-day search performance window.
- **Output Artifact:** An actionable, priority-ranked queue pairing an **Action Priority Score** with explicit **Reason Codes** and concrete **Content Archetype Actions**.
- **Action Taken:** Human editorial triage—dispatching copywriters to rewrite underperforming snippets, subject matter experts to substantively expand striking-distance articles, or technical leads to audit cannibalization.
- **Cost of Wrong Calls:**
  - *False Positive:* Wasting 1.5 to 4.0 hours of expensive editorial time updating content that was structurally sound or experiencing seasonal variance.
  - *False Negative:* Allowing high-equity organic URLs to silently slip past SERP Page 1 into irrecoverable ranking collapse.

Machine learning solves this where fixed rules fail because decay is driven by non-linear interactions across impression velocity, rank position, CTR deficits, and user engagement that simple threshold rules cannot untangle.

---

## 2. Data Safety & Provenance

### Dataset Architecture
Experiments were conducted using FlyRank's enterprise Search Intelligence repository:
1. **Warehouse Release (`hf://datasets/FlyRank/internship-warehouse`):** 78.8M+ daily performance event records across 104 pseudonymized enterprise domains spanning 17 months (January 2025 to June 2026).
2. **Benchmark Production Slice (`data/raw/content_refresh_anonymized.csv`):** A curated 30,000-row cross-client modeling dataset covering 32 client domains, capturing trailing 90-day performance partitioned into recent (last 30 days) and prior (days 31–60 back) comparison windows.

### Deliberate Column Exclusions & Leakage Defense
- **Label Leakage Prevention:** The target variable `is_declining_label` is mathematically defined as `(trend_direction == "down")`, which is calculated directly from `trend_pct = (impressions_last_30d - impressions_prev_30d) / impressions_prev_30d`. Therefore, both `trend_direction` and `trend_pct` were strictly excluded from model feature sets.
- **Identifier Masking:** `client_id` and `content_id` are pseudonymous hash strings used exclusively for grouping and validation splits; neither was ever passed to training matrices.
- **Product Flag Separation:** FlyRank's pre-existing heuristic flags (`health_score`, `quick_win_flag`, `needs_attention`) represent operational outputs. Including them as model inputs creates tautological circularity; they were utilized solely as external baseline comparisons.
- **Privacy Assurance:** Zero raw URLs, client brand names, or private user search queries exist in the repository or report.

---

## 3. Baseline Specification

Before training complex machine learning models, we formalized FlyRank's existing heuristic operational logic into a transparent **Heuristic Action Score** (ML-07):

$$\text{Action Score} = 0.40 \cdot S_{\text{staleness}} + 0.35 \cdot S_{\text{ctr\_gap}} + 0.25 \cdot S_{\text{volume}}$$

- **Staleness Component ($S_{\text{staleness}}$):** Min-max normalized `days_since_last_update` capped at 180 days.
- **CTR Gap Component ($S_{\text{ctr\_gap}}$):** Measured gap between observed CTR and expected SERP position curve benchmarks.
- **Volume Component ($S_{\text{volume}}$):** Log-scaled search impressions over the trailing 90-day window.

On the sealed client-holdout test set (2,325 rows across 6 unseen client domains with a 39.1% decay base rate), this baseline achieved:
- **Precision@20:** 25.0%
- **Precision@50:** 22.0%
- **Precision@100:** 29.0%
- **ROC-AUC:** 0.6262
- **PR-AUC:** 0.4700

While the baseline surfaces high-volume pages, its linear additive structure fails to discriminate subtle non-linear trajectory decay, flagging pages that are old but ranking stably.

---

## 4. Model / Analysis

### Feature Engineering
To capture decay without future leakage, we engineered seven backward-looking signals:
1. `clicks_last_30d`: Recent organic search clicks.
2. `impressions_last_30d`: Recent search impression volume.
3. `decay_ratio_lag`: Lagged ratio of impressions `(impressions_last_30d + 1) / (impressions_prev_30d + 1)`.
4. `position_clean`: Cleaned average rank position, imputing `avg_position == 0` (unranked) to rank 50.0.
5. `ctr`: Trailing click-through rate percentage.
6. `sessions_last_30d`: GA4 website sessions over the recent window.
7. `bounce_proxy`: `100.0 - engagement_rate`, measuring traffic bounce tendency.

### Model Architecture
We evaluated four candidate algorithms: Logistic Regression (L2 penalized), Decision Tree (depth=5), Random Forest (100 estimators), and Gradient Boosted Decision Trees (GBDT, 100 estimators, max_depth=4, learning_rate=0.08). GBDT was selected as the final production champion for its superior ability to handle non-linear SERP interactions and tabular skew.

---

## 5. Evaluation & Validation Audit

### Honest Client-Holdout Validation Protocol
To eliminate domain identity memorization and cross-site feature leakage, we partitioned the 32 client domains into:
- **Training Set:** 26 client domains (27,675 rows, 55.5% decay base rate).
- **Sealed Holdout Set:** 6 client domains (2,325 rows, 39.1% decay base rate).

### Benchmark Comparison (Evaluated on Unseen Clients)

| Model Architecture | PR-AUC | ROC-AUC | P@20 | P@50 | P@100 | Accuracy | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Hand Rule (W4)** | 0.4700 | 0.6262 | 25.0% | 22.0% | 29.0% | 61.1% | 0.3301 |
| **Logistic Regression** | 0.5280 | 0.7005 | 45.0% | 34.0% | 45.0% | 67.4% | 0.5703 |
| **Decision Tree (depth=5)** | 0.5753 | 0.7415 | 80.0% | 68.0% | 65.0% | 67.7% | 0.6339 |
| **Random Forest** | 0.6453 | 0.7577 | 85.0% | 84.0% | 80.0% | 66.7% | 0.6325 |
| **Gradient Boosting (GBDT)** | **0.6772** | **0.7753** | **95.0%** | **86.0%** | **85.0%** | **67.4%** | **0.6465** |

### The Generalization Gap
Our validation audit uncovered an essential methodology insight:
- **Naive Random Split:** GBDT achieved PR-AUC = 0.7821.
- **Honest Client Split:** GBDT achieved PR-AUC = 0.6453 (Random Forest) / 0.6772 (GBDT).
The **0.1368 PR-AUC drop** quantifies domain-memorization reliance in naive evaluations, proving the absolute necessity of client-level holdout validation.

### Error Analysis
- **False Positives (14.0% in Top 50):** Driven primarily by high-volume seasonal pillar URLs experiencing mild cyclic dips that the model mistook for structural decay.
- **False Negatives:** URLs with thin impression baselines (<50 impressions/30d) undergoing severe percentage declines that did not carry sufficient aggregate volume signal.

---

## 6. Interpretation

### Feature Importance
Permutation importance and tree Gini importance identify:
1. `decay_ratio_lag` (38.4% importance): The strongest indicator of forward trajectory.
2. `impressions_last_30d` (24.1% importance): Separates meaningful decay from stochastic low-volume noise.
3. `position_clean` (16.7% importance): Differentiates striking-distance URLs from irrecoverably deep pages.
4. `ctr` (11.2% importance): Highlights title/snippet mismatch on visible impressions.

### Negative Results & Editorial Selection Surprises
In auditing the FlyRank Research Paper's claim that mature refreshed content generates "57x more impressions," we proved this is heavily confounded by editorial selection bias: editors preferentially refresh URLs with 3x higher word count (2,812 vs 970 words) and proven historical authority. Refreshing content does not causally produce a 57x explosion on arbitrary pages.

---

## 7. Ranked Recommendations (Content Action Playbook)

We operationalize model probabilities into a structured **Content Action Playbook**:

$$\text{Action Priority Score} = \hat{P}(\text{Decay}) \times \left(0.5 + 0.5 \times \frac{\ln(1 + \text{impressions}_{30d})}{\max \ln(1 + \text{impressions}_{30d})}\right)$$

| Archetype | Recommended Action | Reason Code | Portfolio Share | Editorial Workload |
| :--- | :--- | :--- | :---: | :---: |
| **PAGE_1_CTR_DEFICIT** | `TITLE_SNIPPET_OPTIMIZATION` | `HIGH_IMPRESSION_SUB_BENCHMARK_CTR` | 20.0% (5,997) | 0.25 hrs/URL |
| **STRIKING_DISTANCE_DECAY**| `REFRESH_AUTHORITY_EXPANSION`| `STRIKING_DISTANCE_SLIPPING` | 11.1% (3,322) | 1.50 hrs/URL |
| **STALE_EVERGREEN_SLIP** | `COMPREHENSIVE_CONTENT_REFRESH`| `MATURE_EVERGREEN_STALENESS` | 7.4% (2,207) | 4.00 hrs/URL |
| **RAPID_TRAFFIC_EROSION** | `TECHNICAL_AND_CONTENT_AUDIT` | `SEVERE_FORWARD_DECAY_VELOCITY` | 9.9% (2,958) | 2.00 hrs/URL |
| **STABLE_CORE_PERFORMER** | `MONITOR_AND_PROTECT` | `HEALTHY_RANKING_PRESERVE_INTEGRITY` | 9.8% (2,948) | 0.00 hrs/URL |
| **NO_GO_INSUFFICIENT_VOL**| `DO_NOT_AUTOMATE` | `LOW_VOLUME_HIGH_NOISE` | 29.6% (8,880) | 0.00 hrs/URL |
| **NO_GO_MANUAL_EDITORIAL**| `MANDATORY_HUMAN_AUDIT` | `HIGH_SENSITIVITY_OR_ANOMALOUS` | 1.5% (450) | 1.00 hrs/URL |

- **Top-100 Queue Editorial Investment:** Triage of the top 100 actionable URLs requires approximately **95.0 total editorial hours**.
- **No-Go Automation Rules:** Prohibits autonomous CMS publishing, automated 410 URL deletions, and automated rewriting of high-stakes comparison content.

---

## 8. Reproducibility

### Execution Instructions
All results, models, figures, and receipts can be regenerated from a clean checkout:

```bash
git clone https://github.com/jaineshchaurasiya20/FlyRank_Ml_Assignment.git
cd FlyRank_Ml_Assignment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### Reproducible Notebook Trajectory
1. [w03_data_contract.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/w03_data_contract.ipynb): Verifies warehouse grain, row counts, and schema contracts.
2. [w04_baseline_score.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/w04_baseline_score.ipynb): Builds and evaluates the heuristic rule baseline.
3. [w05_model.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/w05_model.ipynb): Trains and evaluates Logistic Regression, Trees, Forests, and GBDT.
4. [w06_validation_audit.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/w06_validation_audit.ipynb): Audits research paper findings, tests client-holdout generalization, and proves leakage defense.
5. [w07_action_playbook.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/w07_action_playbook.ipynb): Translates predictions into the Content Action Playbook, generating figures and receipts.
6. [capstone.ipynb](file:///d:/Desktop/FlyRank%20AI%20Internship/FlyRank_Ml_Assignment/work/notebooks/capstone.ipynb): End-to-end synthesis notebook mirroring the deployed research paper.

- **Fixed Random Seed:** `42` across all partitioners and models.
- **Auditable Receipts:** Committed JSON metrics in `work/outputs/` (`baseline_metrics.json`, `model_comparison_metrics.json`, `validation_audit_receipt.json`, `action_playbook_summary.json`).

---

## 9. Acknowledgments & Data Credit

This research was conducted as part of the **FlyRank AI Machine Learning Internship (Summer 2026)**. 

Built on the **FlyRank ML Internship dataset**, provided by [FlyRank AI](https://flyrank.ai/). We thank the FlyRank engineering team for providing access to the pseudonymized enterprise search warehouse and benchmark releases.
