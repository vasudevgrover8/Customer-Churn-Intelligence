# Customer Churn Intelligence

A practical workflow combining statistical analysis, customer segmentation, CatBoost, probability calibration, SHAP explanations, and customer prioritization.

**The highest-risk 20% of held-out customers contained 52.14% of observed churners—approximately 2.61 times random selection.**

[View on Kaggle](https://www.kaggle.com/code/vasudevgrover/customer-churn-intelligence) · [Read the notebook](customer-churn-intelligence.ipynb)

## Practical question

When customer outreach capacity is limited, which customers should be reviewed first?

This project investigates customer characteristics and segments, predicts churn risk, explains model predictions, and measures risk concentration at predefined contact capacities. It does not establish that an intervention would prevent churn.

## Dataset

- Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), shared by BlastChar on Kaggle.
- 7,043 customers from a fictional telecom company.
- 1,869 recorded churners: 26.54% of the sample.
- 11 blank `TotalCharges` entries retained as missing values rather than assumed to be zero.
- Dataset obtained separately from its source under its original terms.

## Workflow

1. Audit data quality, missingness, duplicates, numeric consistency, and target labels.
2. Explore customer behavior and quantify associations using statistical tests, effect sizes, and confidence intervals.
3. Discover and profile four exploratory customer clusters, assessing assignment stability with preprocessing held fixed.
4. Compare a prevalence benchmark, logistic regression, and CatBoost.
5. Fit a sigmoid calibration mapping using training out-of-fold raw scores and assess validation probability quality.
6. Evaluate customer prioritization using precision, churn capture, and lift.
7. Explain CatBoost raw predictions using native SHAP contributions.
8. Apply the fixed models and calibration choice to the held-out test set.

Customer identifiers and churn labels are excluded from clustering and predictive inputs. Cluster assignments are also excluded from predictive inputs.

## Evaluation design

| Partition | Customers | Purpose |
|---|---:|---|
| Training | 4,225 | Exploration, segmentation, model fitting, and out-of-fold calibration inputs |
| Validation | 1,409 | Model and calibration selection |
| Test | 1,409 | Final evaluation |

Splits are stratified by churn, using random seed 42. The random split assesses held-out customers from this sample rather than future customer cohorts.

## Held-out test results

The sigmoid-calibrated CatBoost model selected during validation achieved:

| Metric | Result |
|---|---:|
| ROC AUC | 0.84782 |
| Average precision | 0.67166 |
| Log loss | 0.41325 |
| Brier score | 0.13434 |

Calibration changed probability quality only marginally: test log loss changed from 0.41333 to 0.41325, while Brier score changed from 0.13431 to 0.13434. Ranking metrics remained unchanged at the reported precision.

### Customer prioritization

Customers are ranked by predicted churn probability. Precision is the observed churn proportion within the selected group. Churn capture is the fraction of all observed test churners included. Lift is precision divided by test churn prevalence.

| Contact capacity | Precision | Churn capture | Lift |
|---|---:|---:|---:|
| Top 5% | 83.10% | 15.78% | 3.13× |
| Top 10% | 76.60% | 28.88% | 2.89× |
| Top 20% | 69.15% | 52.14% | 2.61× |
| Top 30% | 59.81% | 67.65% | 2.25× |
| Top 50% | 45.67% | 86.10% | 1.72× |

At the nominal 20% capacity, 282 selected customers included 195 of the 374 observed test churners. Selection counts are rounded upward to whole customers.

These results measure risk concentration, not retention effectiveness. No contact capacity is declared optimal because customer value, intervention costs, and treatment effectiveness are unavailable.

## Explainability

Contract type and tenure have the largest mean absolute SHAP contributions across the validation customers examined, followed by monthly charges and service-related characteristics.

The notebook verifies that explanation inputs reproduce the evaluated CatBoost probabilities and that the baseline plus all SHAP contributions reconstructs the raw prediction. Contributions explain the underlying CatBoost raw score; they are not percentage-point changes in calibrated probability or causal effects.

## Run the notebook

### On Kaggle

1. Open the [published notebook](https://www.kaggle.com/code/vasudevgrover/customer-churn-intelligence) and create an editable copy in your account.
2. Ensure the [source dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) is attached.
3. Run all cells in order.

The loading cells search `/kaggle/input` for `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Locally

Clone the repository and install the directly used third-party packages plus a notebook interface:

```bash
git clone https://github.com/vasudevgrover8/Customer-Churn-Intelligence.git
cd Customer-Churn-Intelligence
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn catboost ipython jinja2 jupyterlab
```

Download the source CSV into a local `data` directory. In the notebook's CSV-discovery cell, replace:

```python
Path("/kaggle/input").rglob("*.csv")
```

with:

```python
Path("data").rglob("*.csv")
```

Launch JupyterLab and run the notebook from top to bottom:

```bash
jupyter lab
```

Local setup instructions are inferred from the notebook imports; a fresh local execution has not been performed for this README. Exact package versions are not pinned, so version changes may affect compatibility or numerical results. SHAP contributions use CatBoost's native API; a separate `shap` installation is not required.

## Limitations and next steps

- Fictional-company data does not establish real-world applicability.
- A random split does not demonstrate temporal generalization.
- Test results are point estimates from one split, without uncertainty intervals or external validation.
- Segments depend on feature choices and preprocessing; the stability assessment holds preprocessing fixed.
- Statistical associations and model explanations do not establish causation.
- Risk concentration does not measure retention effectiveness or financial return.

A practical extension would define a prediction date and future churn horizon, validate on later cohorts, and assess retention interventions through a controlled experiment.

## License and author

Project code is available under the [MIT License](LICENSE). The dataset remains subject to its original terms; this license does not relicense the dataset.

**Vasudev Grover** · [GitHub](https://github.com/vasudevgrover8) · [Kaggle](https://www.kaggle.com/vasudevgrover)
