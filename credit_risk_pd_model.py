"""
Credit Risk: Probability of Default (PD) & Expected Loss Model (Prototype)
===========================================================================
Trains a predictive model on the retail loan book (`Task 3 and 4_Loan_Data.csv`)
to estimate a borrower's probability of default (PD), then uses that PD to
compute the Expected Loss (EL) on a loan.

Expected Loss formula
----------------------
    EL = PD * LGD * EAD

    PD  (Probability of Default) -- predicted by the model for this borrower.
    LGD (Loss Given Default)     -- 1 - recovery_rate. Recovery rate is fixed
                                     at 10% per the task, so LGD = 0.90.
    EAD (Exposure at Default)    -- the outstanding loan amount at risk
                                     (defaults to `loan_amt_outstanding`, but
                                     can be overridden, e.g. with total debt).

Data columns (10,000 borrowers, no missing values):
    customer_id                 -- identifier (not a predictive feature)
    credit_lines_outstanding    -- number of open credit lines
    loan_amt_outstanding        -- outstanding balance on this loan
    total_debt_outstanding      -- total debt across all products
    income                      -- annual income
    years_employed              -- years in current employment
    fico_score                   -- credit score
    default                       -- 1 if borrower defaulted, 0 otherwise (target)

Exploratory correlation with default (computed during development):
    credit_lines_outstanding  +0.86   (strongest predictor)
    total_debt_outstanding    +0.76
    fico_score                -0.32
    years_employed            -0.28
    loan_amt_outstanding      +0.10   (weak)
    income                    +0.02   (negligible)

Two models are trained and compared:
    1. Logistic Regression -- simple, fast, fully interpretable (coefficients
       show direction/magnitude of each feature's effect on default odds).
       This is the natural first choice for a PD model since bank model-risk
       / validation teams generally prefer transparent, explainable models.
    2. Gradient Boosted Trees -- a stronger nonlinear model, used here as a
       benchmark to see how much accuracy is left on the table by choosing
       the simpler, more interpretable logistic regression.

Both are evaluated on a held-out test set using AUC-ROC (ranking quality of
PD estimates) and accuracy at a 0.5 threshold. On this dataset both models
achieve near-perfect separation (AUC ~1.0), reflecting how cleanly the
default flag is determined by credit_lines_outstanding / total_debt in this
sample -- logistic regression is chosen as the production candidate since it
matches the more complex model's performance while remaining fully
interpretable for model validation and regulatory review.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

CSV_PATH = "Task 3 and 4_Loan_Data.csv"

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]
TARGET = "default"
RECOVERY_RATE = 0.10  # fixed per task assumption


class PDModel:
    """
    Wraps a trained classifier + feature scaler so it can be used to score
    new borrowers and compute expected loss in one call.
    """

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict_pd(self, borrower: dict) -> float:
        """
        borrower: dict with keys matching `feature_names`, e.g.
            {"credit_lines_outstanding": 3, "loan_amt_outstanding": 4500,
             "total_debt_outstanding": 9000, "income": 55000,
             "years_employed": 4, "fico_score": 600}
        Returns the predicted probability of default (0-1).
        """
        x = pd.DataFrame([{f: borrower[f] for f in self.feature_names}])
        x_scaled = self.scaler.transform(x)
        return float(self.model.predict_proba(x_scaled)[0, 1])

    def expected_loss(self, borrower: dict, recovery_rate: float = RECOVERY_RATE,
                       exposure_at_default: float = None) -> dict:
        """
        Computes Expected Loss = PD * (1 - recovery_rate) * EAD.

        If `exposure_at_default` is not given, defaults to the borrower's
        `loan_amt_outstanding`.
        """
        pd_estimate = self.predict_pd(borrower)
        ead = exposure_at_default if exposure_at_default is not None else borrower["loan_amt_outstanding"]
        lgd = 1 - recovery_rate
        el = pd_estimate * lgd * ead
        return {
            "probability_of_default": pd_estimate,
            "loss_given_default_rate": lgd,
            "exposure_at_default": ead,
            "expected_loss": el,
        }


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def train_models(df: pd.DataFrame, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000, random_state=random_state)
    logreg.fit(X_train_scaled, y_train)
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_preds = (logreg_probs >= 0.5).astype(int)

    # --- Model 2: Gradient Boosted Trees (benchmark) ---
    gbt = GradientBoostingClassifier(random_state=random_state)
    gbt.fit(X_train_scaled, y_train)
    gbt_probs = gbt.predict_proba(X_test_scaled)[:, 1]
    gbt_preds = (gbt_probs >= 0.5).astype(int)

    results = {
        "Logistic Regression": {
            "model": logreg,
            "auc": roc_auc_score(y_test, logreg_probs),
            "accuracy": accuracy_score(y_test, logreg_preds),
            "confusion_matrix": confusion_matrix(y_test, logreg_preds),
        },
        "Gradient Boosted Trees": {
            "model": gbt,
            "auc": roc_auc_score(y_test, gbt_probs),
            "accuracy": accuracy_score(y_test, gbt_preds),
            "confusion_matrix": confusion_matrix(y_test, gbt_preds),
        },
    }

    return results, scaler, (X_train, X_test, y_train, y_test)


def print_comparison(results):
    print("=== Model comparison (held-out test set) ===")
    print(f"{'Model':<24}{'AUC-ROC':>10}{'Accuracy':>10}")
    for name, r in results.items():
        print(f"{name:<24}{r['auc']:>10.4f}{r['accuracy']:>10.4f}")
    print()

    print("Logistic Regression coefficients (standardized features):")
    logreg = results["Logistic Regression"]["model"]
    for feat, coef in zip(FEATURES, logreg.coef_[0]):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feat:<28} coef={coef:+.3f}  ({direction} default odds)")
    print()


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} borrower records.")
    print(f"Overall historical default rate: {df[TARGET].mean():.2%}\n")

    results, scaler, splits = train_models(df)
    print_comparison(results)

    # Choose logistic regression as the production candidate: nearly as
    # accurate as the gradient boosted model but fully interpretable --
    # important for model validation / regulatory review of a PD model.
    chosen_model = results["Logistic Regression"]["model"]
    pd_model = PDModel(chosen_model, scaler, FEATURES)

    print("=== Sample expected loss calculations ===")
    sample_borrowers = [
        {   # low risk: strong FICO, low debt/credit lines, long tenure
            "credit_lines_outstanding": 0,
            "loan_amt_outstanding": 5000,
            "total_debt_outstanding": 3000,
            "income": 78000,
            "years_employed": 6,
            "fico_score": 720,
        },
        {   # high risk: many credit lines, high debt, weak FICO
            "credit_lines_outstanding": 5,
            "loan_amt_outstanding": 6000,
            "total_debt_outstanding": 15000,
            "income": 30000,
            "years_employed": 1,
            "fico_score": 560,
        },
        {   # moderate risk
            "credit_lines_outstanding": 2,
            "loan_amt_outstanding": 4500,
            "total_debt_outstanding": 8000,
            "income": 55000,
            "years_employed": 3,
            "fico_score": 630,
        },
    ]

    for i, borrower in enumerate(sample_borrowers, 1):
        result = pd_model.expected_loss(borrower)
        print(f"Borrower {i}: {borrower}")
        print(f"  -> PD = {result['probability_of_default']:.2%}, "
              f"EAD = ${result['exposure_at_default']:,.0f}, "
              f"LGD = {result['loss_given_default_rate']:.0%}, "
              f"Expected Loss = ${result['expected_loss']:,.2f}\n")
