"""Generate a realistic, reproducible customer churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_rows)
    monthly_charges = np.clip(rng.normal(78, 28, n_rows), 20, 180).round(2)
    total_charges = np.maximum(monthly_charges * tenure + rng.normal(0, 250, n_rows), 20).round(2)
    support_tickets = rng.poisson(1.8, n_rows)
    usage_drop_pct = np.clip(rng.normal(8, 16, n_rows), -35, 75).round(1)
    late_payments = rng.poisson(0.7, n_rows)
    satisfaction = np.clip(np.round(rng.normal(7.1, 1.8, n_rows), 1), 1, 10)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["Electronic check", "Bank transfer", "Credit card", "Mailed check"], n_rows, p=[0.34, 0.24, 0.28, 0.14])
    internet_service = rng.choice(["Fiber optic", "DSL", "None"], n_rows, p=[0.45, 0.42, 0.13])
    tech_support = rng.choice(["Yes", "No"], n_rows, p=[0.35, 0.65])
    senior_citizen = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    dependents = rng.choice([0, 1], n_rows, p=[0.70, 0.30])
    autopay = rng.choice([0, 1], n_rows, p=[0.58, 0.42])

    logit = (
        -2.8 + 0.045 * (monthly_charges - 70) - 0.035 * tenure
        + 0.20 * support_tickets + 0.025 * usage_drop_pct + 0.30 * late_payments
        - 0.23 * (satisfaction - 7) + 0.75 * (contract_type == "Month-to-month")
        + 0.32 * (payment_method == "Electronic check") + 0.34 * (internet_service == "Fiber optic")
        - 0.40 * (tech_support == "Yes") + 0.28 * senior_citizen - 0.25 * dependents
        - 0.38 * autopay
    )
    # A modestly amplified signal keeps the synthetic benchmark learnable while
    # preserving overlap between churners and non-churners.
    churn_probability = 1 / (1 + np.exp(-(1.35 * logit)))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_rows + 1)],
        "tenure_months": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
        "support_tickets_90d": support_tickets, "usage_drop_pct": usage_drop_pct,
        "late_payments_12m": late_payments, "satisfaction_score": satisfaction,
        "contract_type": contract_type, "payment_method": payment_method,
        "internet_service": internet_service, "tech_support": tech_support,
        "senior_citizen": senior_citizen, "dependents": dependents, "autopay": autopay,
        "churn": churn,
    })


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv"
    output.parent.mkdir(exist_ok=True)
    generate_dataset().to_csv(output, index=False)
    print(f"Wrote {len(generate_dataset())} rows to {output}")


FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets_90d", "usage_drop_pct",
    "late_payments_12m", "satisfaction_score", "contract_type", "payment_method", "internet_service",
    "tech_support", "senior_citizen", "dependents", "autopay",
]
TARGET = "churn"
ID_COL = "customer_id"
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service", "tech_support"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]


def mitigation_actions(row: pd.Series) -> list[str]:
    actions = []
    if row.get("contract_type") == "Month-to-month": actions.append("Offer a 12-month plan with a loyalty incentive")
    if row.get("satisfaction_score", 10) <= 6: actions.append("Schedule a customer-success check-in within 48 hours")
    if row.get("usage_drop_pct", 0) >= 15: actions.append("Investigate usage decline and provide an adoption session")
    if row.get("support_tickets_90d", 0) >= 3: actions.append("Escalate unresolved support themes to service recovery")
    if row.get("late_payments_12m", 0) >= 2: actions.append("Offer billing assistance and enable autopay")
    if row.get("tech_support") == "No": actions.append("Offer a complimentary technical-support trial")
    if not actions: actions.append("Maintain proactive engagement and monitor monthly risk")
    return actions


def risk_band(probability: float) -> str:
    if probability >= 0.70: return "High"
    if probability >= 0.40: return "Medium"
    return "Low"


def scorecard(row: pd.Series, probability: float) -> dict:
    return {"customer_id": row.get("customer_id", "New customer"), "risk_probability": round(float(probability), 4),
            "risk_band": risk_band(probability), "priority": "Immediate" if probability >= 0.70 else ("This week" if probability >= 0.40 else "Monitor"),
            "recommended_actions": mitigation_actions(row)}


def generate_scorecards(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    cards = [scorecard(row, p) for (_, row), p in zip(df.iterrows(), probabilities)]
    out = pd.DataFrame(cards)
    out["recommended_actions"] = out["recommended_actions"].apply(lambda x: " | ".join(x))
    return out
