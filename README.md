![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)
# Predicting Customer Churn

## Project 2  Machine Learning: From Data to Predictions

This project looks at a fairly simple business question:

> **Which customers are most likely to leave, so a retention team can contact them before they do?**

I used the IBM Telco Customer Churn dataset to build and evaluate a supervised machine learning model that predicts whether an existing telecom customer will churn.

The aim was not simply to get the highest possible score. I wanted a model whose performance I could explain, compare honestly against a baseline, and potentially use as a practical prioritisation tool.

---

## The data

The dataset contains **7,043 telecom customers**, with one row per customer.

It includes:

- customer demographics
- account tenure
- contract type
- billing and payment information
- internet and phone services
- monthly and total charges
- whether the customer churned

The target is:

```text
Churn
```

with:

```text
Yes = customer churned
No  = customer stayed
```

Around **26.5% of customers churned**, so the target is moderately imbalanced.

### Source and licence

Telco customer churn sample data, IBM, from
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d).

Licensed under the **Apache 2.0 licence**.

---

## Business framing

The intended user is a **customer retention team** with limited time and budget.

The model is not intended to automatically decide what happens to a customer. Its job is to help answer:

> **Who should we contact first?**

A second, simpler question supports that decision:

> **Which existing customer characteristics are most strongly associated with churn?**

That second question gives a useful sense-check on whether the final model has learned patterns that make business sense.

For this use case, I considered a **false negative** more costly than a false positive.

A false negative means the model predicts that a customer will stay when they actually leave. That removes the opportunity for the business to intervene.

A false positive means the business contacts someone who would probably have stayed anyway. That wastes some retention time or budget, but is less damaging than missing a genuine churner.

For that reason, I chose **recall for the churn class** as the main model metric.

I also monitored precision, F1 and accuracy so that improving recall did not simply turn into predicting churn for everyone.

---

## Data preparation

A few issues needed dealing with before modelling.

### `TotalCharges`

`TotalCharges` was stored as text rather than a numeric column.

There were **11 blank values**, all belonging to customers with a tenure of 0 months. These were new customers who had not yet accumulated total charges.

I converted the column to numeric and filled these values with `0`.

### `customerID`

`customerID` uniquely identifies individual customers but contains no meaningful information for predicting churn.

It was removed before modelling.

### Train/test split

The data was split once into:

- **80% training data**
- **20% test data**

using:

```python
random_state=42
stratify=y
```

Stratification kept the proportion of churners approximately equal in both sets.

The test set was then left untouched until the final model had been selected.

---

## Exploratory analysis

Some clear churn patterns appeared before any model was built.

Customers who churned tended to have:

- shorter tenure
- month-to-month contracts
- fibre optic internet
- electronic check payments
- higher monthly charges

Customers with longer tenure and longer contracts were considerably less likely to churn.

These relationships were useful later when interpreting the final model, but they should not be treated as proof of causation.

---

## Baseline

Before building a real model, I used a `DummyClassifier` that always predicts the majority class.

Because the majority of customers do not churn, this model predicted `No` for everyone.

Its churn recall was:

```text
0.000
```

That gives a deliberately low but useful benchmark: any genuine model should be substantially better at finding churners.

---

## Preprocessing pipeline

All preprocessing that learns from the data was kept inside a scikit-learn pipeline.

Numeric features:

```text
tenure
MonthlyCharges
TotalCharges
```

were:

1. median-imputed where required
2. standardised with `StandardScaler`

Categorical features were:

1. imputed using the most frequent category
2. encoded using `OneHotEncoder`

Keeping preprocessing inside the pipeline means that during cross-validation each fold is processed using only information from its own training data.

This avoids data leakage.

---

## Feature engineering

I tested an additional categorical `TenureGroup` feature.

The original Logistic Regression achieved mean cross-validated recall of:

```text
0.544
```

With `TenureGroup`:

```text
0.536
```

The engineered feature made performance slightly worse, so I did not keep it.

There was no point adding complexity just because I could.

---

## Models compared

I compared three different model families using the **same five cross-validation folds**:

| Model | Mean CV Recall |
|---|---:|
| Logistic Regression | 0.544 |
| Decision Tree | 0.500 |
| Random Forest | 0.486 |
| Dummy baseline | 0.000 |

Logistic Regression performed best before tuning.

Decision Tree was reasonably close, so I tuned both Logistic Regression and Decision Tree using `GridSearchCV`.

---

## Model tuning

### Logistic Regression

Best parameters:

```python
C=1
class_weight="balanced"
```

Best cross-validated recall:

```text
0.801 ± 0.038
```

### Decision Tree

Best parameters:

```python
class_weight="balanced"
max_depth=5
min_samples_leaf=10
```

Best cross-validated recall:

```text
0.781 ± 0.035
```

The two tuned models are close. Logistic Regression is only about **0.020** higher on mean recall, and that gap is smaller than the normal variation between folds. I would not claim that it massively outperformed the Decision Tree. It wins narrowly on recall, and it is also much easier to explain and use in practice. That is enough for me to choose it.

The big improvement came from `class_weight="balanced"`, not from complicated tuning. The best Logistic Regression kept the default `C=1`; giving the less common churn class more weight is what moved recall from 0.544 to 0.801.

---

## Classification threshold

I did not assume that the standard probability threshold of `0.50` was automatically correct.

Using cross-validated probabilities from the training set, I compared precision, recall and F1 across a range of thresholds.

The business aim was to catch at least **80% of genuine churners**, while keeping precision as high as possible.

A threshold of:

```text
0.50
```

already gave the best balance for that requirement, so I kept it.

This was a decision based on the training data only. The test set was still untouched.

---

# Final test results

The final Logistic Regression pipeline was fitted to the full training set and then evaluated **once** on the untouched test set.

| Metric | Result |
|---|---:|
| Accuracy | 0.738 |
| Precision — Churn | 0.504 |
| Recall — Churn | **0.783** |
| F1 — Churn | 0.614 |

The cross-validated estimate of recall had been:

```text
0.801 ± 0.038
```

The spread is the standard deviation across the same five stratified cross-validation folds used throughout model comparison.

The final test recall was:

```text
0.783
```

a difference of only:

```text
-0.018
```

That is a small drop and it sits comfortably inside the variation I had already seen during cross-validation. In other words, the test result looks stable rather than like I got lucky with one split.

Compared with the baseline recall of **0.000**, the model is clearly finding churners rather than simply benefiting from the fact that most customers stay.

---

## Confusion matrix

On the test set:

| | Predicted Stay | Predicted Churn |
|---|---:|---:|
| **Actually stayed** | 747 | 288 |
| **Actually churned** | 81 | 293 |

The model correctly identified:

```text
293 of 374 churners
```

and missed:

```text
81 churners
```

At the chosen threshold the model flags **581 of the 1,409 test customers** for retention attention. That queue contains 293 genuine churners and **288 customers who actually stayed**.

For a Head of Customer Service, the useful question is no longer just "is the model accurate?" It is **can my team actually work 581 proactive contacts?** If not, I would rank the flagged customers by score and start with the highest-risk cases.

For this use case, I think that is a reasonable trade-off if retention contact is relatively inexpensive.

If contacting customers involved a substantial discount or high operational cost, I would revisit the threshold and accept lower recall in exchange for higher precision.

---

## Where the model gets it wrong

The most confident false negatives were generally customers with:

- long tenure
- one- or two-year contracts
- DSL
- automatic payment methods

In other words, they looked very much like customers who normally stay — but left anyway.

The strongest false positives showed almost the exact opposite pattern:

- very short tenure
- month-to-month contracts
- fibre optic internet
- electronic check payment
- relatively high monthly charges

These customers looked like classic churn risks according to the historical data, but stayed.

That is a useful reminder that a prediction is a probability, not a fact.

Individual customers can still behave differently from people with otherwise similar profiles.

---

## What drives the predictions?

I used both Logistic Regression coefficients and permutation importance to understand the model.

Permutation importance showed the strongest original features were:

1. **tenure**
2. **Contract**
3. **InternetService**
4. **MonthlyCharges**

Tenure was comfortably the most important.

The coefficient analysis gave some extra detail.

Features associated with higher predicted churn included:

- fibre optic internet
- month-to-month contracts
- electronic check payments
- no online security

Features associated with staying included:

- longer tenure
- two-year contracts
- DSL

These results also broadly matched the original exploratory analysis.

The charge-related coefficients need more careful interpretation because `tenure`, `MonthlyCharges` and `TotalCharges` are strongly related to one another.

None of these relationships should be interpreted as proving that a feature **causes** churn.

---

## Business recommendation

I would use this model as a **prioritisation tool for retention outreach**, not as an automatic decision-maker.

It catches around **78% of customers who actually churn**, which makes it useful for narrowing down a large customer base to a higher-risk group.

Its churn precision is around **50%**, however, so not everyone flagged by the model will actually leave.

That means it makes most sense where:

- contacting a customer is relatively cheap
- there is enough retention capacity to tolerate some unnecessary contacts
- the model is used alongside normal customer and commercial judgement

I would not use the prediction alone to automatically issue discounts, change contracts or make decisions that have a significant financial impact.

---

## Limitations

This model only knows what exists in this dataset.

It does **not** contain potentially useful information such as:

- recent complaints
- service outages
- customer satisfaction scores
- competitor offers
- recent support interactions
- reasons given for previous cancellations

It also represents one historical customer population.

The model probabilities have **not been separately calibrated**, so I would use the scores to rank customers rather than tell a manager that an 80% score means the customer has exactly an 80% real-world chance of leaving.

Customer behaviour, pricing, products and competitors change, so model performance should not be assumed to stay the same forever.

In a live business setting I would keep an eye on:

- churn rate
- recall
- precision
- false positive workload
- changes in the customer data going into the model

If those started moving materially, I would review the model rather than assume it would stay useful forever.

---

## Saved model and demo

The complete fitted preprocessing and Logistic Regression pipeline is saved using `joblib`.

The classification threshold is saved alongside it.

Notebook 03 reloads the saved model and predicts churn probabilities for new customer records.

For example, three demonstration customers produced:

| Tenure | Contract | Internet | Payment | Churn probability | Prediction |
|---:|---|---|---|---:|---|
| 3 | Month-to-month | Fibre optic | Electronic check | 89.2% | Churn |
| 60 | Two year | DSL | Bank transfer (automatic) | 2.1% | Stay |
| 18 | One year | Fibre optic | Credit card (automatic) | 62.6% | Churn |

The saved model itself is not committed to GitHub because the `models/` directory is gitignored. It can be recreated by running the notebooks.

---

# Project structure

```text
project-2-ml/
│
├── data/
│   └── raw/                         # Downloaded data - not committed
│
├── models/                          # Generated model - not committed
│
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_pipelines_and_models.ipynb
│   └── 03_evaluation_and_interpretation.ipynb
│
├── src/
│   ├── functions.py
│   └── demo.py                       # Reusable Streamlit scoring helpers
│
├── app.py                           # Streamlit retention decision-support demo
├── alex_telco_churn_presentation.pptx
├── download_data.py
├── requirements.txt
└── README.md
```

---

# How to reproduce the project

Clone the repository and move into the project folder.

```bash
git clone https://github.com/iamlama1982-prog/project-2-ml.git
cd project-2-ml
```

Create or activate your Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

Download the Telco dataset:

```bash
python download_data.py telco
```

This creates:

```text
data/raw/telco_churn.csv
```

Then run the notebooks in order:

```text
1. notebooks/01_eda_and_baseline.ipynb
2. notebooks/02_pipelines_and_models.ipynb
3. notebooks/03_evaluation_and_interpretation.ipynb
```

Notebook 03 recreates and saves the final model before reloading it for the demonstration predictions.

### Run the Streamlit demo

After the model has been created by notebook 03:

```bash
streamlit run app.py
```

The app is built for a **Head of Customer Service / Call Centre** rather than as another EDA dashboard. It provides:

- a ranked retention queue
- a contact-capacity slider that translates predictions into workload
- a back-test of how many known churners that capacity would have reached in the held-out demo data
- CSV upload and prioritised queue export
- live scoring for one customer, including the strongest row-level model signals

Locally, the app loads the saved fitted pipeline from `models/`. For Streamlit Community Cloud, where model artefacts remain intentionally gitignored, it can recreate the exact documented final pipeline from the IBM source data using the same fixed split and model parameters.

---

## Tools used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- joblib
- Streamlit
- Jupyter / VS Code

---

## Presentation and demo

Presentation slides: [`alex_telco_churn_presentation.pptx`](alex_telco_churn_presentation.pptx)

Interactive demo: `app.py` (public Streamlit URL added after deployment).

---

## Final takeaway

The final model is not perfect, and that is rather the point.

It does not tell us with certainty who will leave.

What it does do is identify **around 78% of genuine churners** from information already available to the business, with performance on unseen data very close to what cross-validation predicted.

For a retention team trying to decide **where to look first**, that is useful.
