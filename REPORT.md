# ESG Analytics — Progress Report (Steps 1–6)

**Author:** Vivek Parashar
**Project:** ESG Sustainability Analytics on S&P 500 Companies
**Scope of this update:** Data Cleaning → Quality Validation → Feature Engineering → EDA → Risk Index → Next-Year Forecasting
**Dataset:** `preprocessed_content.csv` — 866 sustainability-report rows × 10 columns (263 companies, 2014–2023). Source: Kaggle ("ESG Sustainability Reports of S&P 500 Companies"). Text, named entities and E/S/G scores are pre-computed in the dataset; my work builds the full analytics, risk and modelling layer on top of it.

---

## 1. Objective

Companies publish 100+ page sustainability reports every year, which are impossible to compare manually. The goal of this project is to turn a pre-scored ESG text dataset into **decision-useful intelligence** — cleaned and validated data, engineered signals, exploratory insights, a company risk-ranking, and a leakage-safe model that forecasts next-year ESG performance. Steps 7–8 (a GenAI copilot and a Streamlit dashboard) are planned next and are not part of this update.

---

## 2. Methodology, Reasoning and Results

Each step below states **what** was done, **why**, and the **result**.

### Step 1 — Data Cleaning
**What:** Dropped a stray pandas index column (`Unnamed: 0`); parsed the `ner_entities` string column into real Python lists using `ast.literal_eval` (not `eval`, to avoid arbitrary-code-execution risk); standardised text (`ticker` upper-cased and stripped) and numeric types (`pd.to_numeric(errors="coerce")`); removed duplicate reports keyed on `ticker + year`.
**Why:** An index column left in place can leak into the model as a fake feature; string-formatted lists cannot be used as lists; duplicate filings (the same company-year filed on two exchanges) would double-count in every downstream statistic.
**Result:** 866 × 10 → **864 × 9** (2 duplicate ABT-2021 filings removed). Clean, correctly-typed dataset.

### Step 2 — Data Quality Validation
**What:** Tested the assumed relationship `total_score = e + s + g`, validated score ranges and years, and checked for missing values.
**Why:** Assumptions must be verified, not trusted. If `total_score` were exactly the sum of the pillars, using the current pillars to predict the total would be circular (data leakage).
**Result (honest finding):** The relationship is **additive and equal-weight** — a linear fit gives coefficients **[1.00, 1.00, 1.00]** with **R² = 0.9993** — but it is **not an exact identity**. Only 40.0% of rows match to two decimals and 77.5% fall within a rounding-level tolerance (0.02); 99.2% are within 1.0, yet **~22% of rows (194) deviate by more than floating-point rounding can explain (max gap 2.77)**. Conclusion: `total_score` is treated as an *approximately-additive, separately-reported* figure, compared with `np.isclose`/tolerance rather than `==`, with large-gap rows flagged as data-quality outliers. No missing values; all scores non-negative; years within 2014–2023.

### Step 3 — Feature Engineering
**What:** Built **18 derived features** in three groups:
- **Text/entity:** `word_count`, `char_count`, `entity_count`, `unique_entities`, `entity_diversity`
- **Composition:** `e_share`, `s_share`, `g_share`, `dominant_pillar`, `pillar_std`
- **Lag/time:** `prev_total`, `prev_e/s/g`, `roll_mean_total`, `yoy_change`, `years_since_2014`, `report_seq`
**Why:** Raw columns are not "model-ready". Text features capture disclosure depth; composition features capture a company's focus (not just size); lag features (built with `shift(1)` inside each company) allow next-year forecasting using **only past information** — the mechanism that keeps the model leakage-free.
**Result:** 9 → **27 columns**, saved as `esg_features.csv`.

### Step 4 — Exploratory Data Analysis
**What:** Four targeted charts — pillar score distributions, pillar correlation heatmap, temporal trend (2014–2023), and top-10 companies by average score.
**Why:** EDA is about insight, not chart count; each chart answers a specific business question.
**Result / insights:** Social is the most-disclosed pillar (mean 10.3) and Environmental the least (mean 5.9); pillars are positively correlated; average disclosure rises over 2014–2023 (regulatory pressure); the top-ranked companies are concentrated in energy/oil & gas (sectors under the most ESG scrutiny). Figures saved in `figures/`.

### Step 5 — ESG Risk Index
**What:** Normalised each pillar to 0–100 (`MinMaxScaler`) and combined them into a weighted index — **Risk Index = 0.40·E + 0.30·S + 0.30·G** — then bucketed into High/Medium/Low and ranked all companies.
**Why:** Executives need a single comparable number and simple bands, not three raw decimals. Environmental is weighted highest because climate is currently the most material ESG risk; the weights are an explicit, configurable business judgment.
**Result:** `risk_index` spans 3.6–66.0; **529 Low, 335 Medium, 0 High** (no company crosses the 66 High threshold — an honest artifact of the scale). All **263 companies ranked**; energy names (CTRA, EOG, CVX, PXD) top the list. Saved as `esg_risk.csv`.

### Step 6 — Next-Year ESG Forecasting (leakage-safe)
**What:** Framed a regression problem — predict next year's `total_score` from **past/structural features only** (`prev_total`, `prev_e/s/g`, `roll_mean_total`, `word_count`, `entity_count`, `years_since_2014`, `report_seq`). Explicitly **excluded** current-year `e/s/g/total` *and* `yoy_change` (since `yoy_change = total − prev_total` hides the target and would give a fake R² = 1.0). Compared Linear Regression, Random Forest and a naive persistence baseline on an 80/20 split (seed 42).
**Why:** A model that only sees the past can actually be deployed; one that peeks at the answer cannot. Beating a naive baseline proves the engineered features carry real signal.
**Result (test set, 603 modelling rows):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **1.59** | **2.23** | **0.913** |
| Random Forest (200) | 1.62 | 2.29 | 0.908 |
| Naive baseline (last = next) | 1.75 | 2.50 | 0.891 |

Both models beat the naive baseline; Linear Regression wins (simplest model generalises best on small, largely-linear data). `prev_total` is by far the strongest predictor. Model saved to `outputs/models/esg_model.pkl`. R² ≈ 0.91 is an **honest** forecasting result, not a leakage-inflated 1.0.

---

## 3. Results Summary

| Step | Output | Key result |
|---|---|---|
| 1 Clean | 864 × 9 dataset | 2 duplicate filings removed, types fixed |
| 2 Validate | Documented assumptions | total ≈ e+s+g (coeffs 1,1,1, R²=0.999); *approximately* additive, not exact |
| 3 Features | 27 columns | 18 engineered (text / composition / leakage-safe lag) |
| 4 EDA | 4 charts | Social highest, Environmental lowest; disclosure rising; energy sector leads |
| 5 Risk Index | `esg_risk.csv` | 263 companies ranked; 40-30-30 weighted index |
| 6 Forecast | `esg_model.pkl` | Leakage-safe Linear Regression, R² 0.913, beats baseline |

---

## 4. Data-Integrity Note
The most important finding of this update is in Step 2: a widely-assumed exact identity (`total = e+s+g`) turned out to be only *approximately* true in the raw data. Rather than hand-wave the gap as "rounding", it is characterised honestly (~22% of rows deviate beyond rounding, up to 2.77) and handled with tolerance-based comparison and outlier flagging. This discipline — verifying assumptions instead of trusting them — underpins the reliability of every later step.

## 5. Next Steps
- **Step 7 — GenAI Copilot (RAG):** natural-language Q&A grounded on the ESG data.
- **Step 8 — Streamlit Dashboard:** interactive scores, trends, risk-ranking and the copilot in one app.
- Optional model upgrades: TF-IDF/FinBERT-based scoring, cross-validation, and sensitivity analysis on the risk weights.

---

*Reproducibility: all results above are produced end-to-end by `ESG_Methodology_Steps_1-6.ipynb` (runs top-to-bottom with zero errors on the Kaggle dataset).*
