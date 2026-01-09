"""
Utility functions for the Fraud Analytics Dashboard
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Path to CSV exports
CSV_EXPORTS = Path(__file__).parent.parent / "content" 

def load_data(dataset_name="df_exp_50.csv"):
    """Load dataset from content"""
    try:
        df = pd.read_csv(CSV_EXPORTS / dataset_name)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

def get_summary_stats(df):
    """Calculate summary statistics for dashboard"""
    if df is None or df.empty:
        return {}
    
    stats = {
        'total_transactions': len(df),
        'fraud_count': int(df['Class'].sum()) if 'Class' in df.columns else 0,
        'fraud_rate': float(df['Class'].mean() * 100) if 'Class' in df.columns else 0,
        'total_amount': float(df['Amount'].sum()) if 'Amount' in df.columns else 0,
        'avg_amount': float(df['Amount'].mean()) if 'Amount' in df.columns else 0,
        'fraud_amount': float(df[df['Class']==1]['Amount'].sum()) if 'Class' in df.columns and 'Amount' in df.columns else 0,
    }
    return stats

def get_temporal_data(df, freq="H"):
    """Get temporal aggregation of transactions"""
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    dff = df.copy()
    dff = dff.loc[dff["timestamp"].notna()].copy()
    if dff.empty:
        return None

    dff["time_bin"] = dff["timestamp"].dt.floor(freq)

    agg_dict = {
        "Class": ["count", "sum", "mean"],  # count: n_tx, sum: n_fraud, mean: fraud rate (0-1)
    }
    has_amount = "Amount" in dff.columns
    if has_amount:
        agg_dict["Amount"] = ["sum", "mean"]

    temporal = dff.groupby("time_bin").agg(agg_dict).reset_index()

    if has_amount:
        temporal.columns = ["timestamp", "count", "fraud_count", "fraud_rate", "total_amount", "avg_amount"]
    else:
        temporal.columns = ["timestamp", "count", "fraud_count", "fraud_rate"]
        temporal["total_amount"] = np.nan
        temporal["avg_amount"] = np.nan

    temporal["fraud_rate"] = temporal["fraud_rate"] * 100  # %
    return temporal.sort_values("timestamp")


def get_country_stats(df, min_tx=100, m=500, top_n=None, sort_by="fraud_rate_smoothed_pct"):
    """
    Country stats for dashboard.

    Returns a DataFrame with:
      - total transactions
      - fraud_count
      - fraud_rate_pct (raw)
      - fraud_rate_smoothed_pct (Bayesian smoothing)
      - total_amount
      - fraud_amount

    Args:
        min_tx: minimum number of transactions to appear
        m: smoothing strength (higher => more pull towards global mean)
        top_n: keep only top N rows after sorting (optional)
        sort_by: one of ["fraud_rate_smoothed_pct","fraud_amount","fraud_count","total"]
    """
    if df is None or df.empty:
        return None
    if "customer_country" not in df.columns or "Class" not in df.columns:
        return None

    dff = df.copy()
    dff["customer_country"] = dff["customer_country"].fillna("Unknown")

    has_amount = "Amount" in dff.columns

    agg_dict = {"Class": ["count", "sum"]}
    if has_amount:
        agg_dict["Amount"] = ["sum"]

    cs = dff.groupby("customer_country").agg(agg_dict).reset_index()

    if has_amount:
        cs.columns = ["country", "total", "fraud_count", "total_amount"]
    else:
        cs.columns = ["country", "total", "fraud_count"]
        cs["total_amount"] = np.nan

    # Fraud amount (real impact): sum Amount where Class==1 by country
    if has_amount:
        fraud_amount = (
            dff.loc[dff["Class"] == 1]
            .groupby("customer_country")["Amount"]
            .sum()
            .rename("fraud_amount")
            .reset_index()
            .rename(columns={"customer_country": "country"})
        )
        cs = cs.merge(fraud_amount, on="country", how="left")
        cs["fraud_amount"] = cs["fraud_amount"].fillna(0.0)
    else:
        cs["fraud_amount"] = np.nan

    global_rate = float(dff["Class"].mean())  # 0-1
    cs["fraud_rate"] = cs["fraud_count"] / cs["total"]
    prior_strength = m  
    cs["fraud_rate_smoothed"] = (cs["fraud_count"] + prior_strength * global_rate) / (cs["total"] + prior_strength)

    cs["fraud_rate_pct"] = cs["fraud_rate"] * 100
    cs["fraud_rate_smoothed_pct"] = cs["fraud_rate_smoothed"] * 100

    cs = cs.loc[cs["total"] >= min_tx].copy()

    if sort_by not in cs.columns:
        sort_by = "fraud_rate_smoothed_pct"

    cs = cs.sort_values(sort_by, ascending=False)

    if top_n is not None:
        cs = cs.head(int(top_n))

    return cs


def get_model_metrics(dataset_key=None):
    """
    Get model performance metrics from saved JSON results.

    Returns dict with aligned arrays per metric. Missing metrics = None.
    """
    import json
    
    json_path = Path(__file__).parent / "data" / "model_results.json"
    
    def _empty():
        return {
            "models": [],
            "f1_scores": [],
            "f2_scores": [],
            "precision": [],
            "recall": [],
            "auc_pr": [],
            "roc_auc": [],
        }
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Warning: could not load model results at {json_path}: {e}")
        return _empty()
    
    if dataset_key is None:
        dataset_key = results_data.get("best_model", {}).get("dataset")
    
    dataset_info = results_data.get("datasets", {}).get(dataset_key, {})
    if not dataset_info:
        return _empty()
    
    models_dict = dataset_info.get("models", {})
    model_order = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogisticRegression"]
    
    out = {
        "models": [],
        "f1_scores": [],
        "f2_scores": [],
        "precision": [],
        "recall": [],
        "auc_pr": [],
        "roc_auc": [],
    }
    
    for model_name in model_order:
        if model_name not in models_dict:
            continue
        mm = models_dict[model_name]
        # Format model names for display
        display_name = model_name
        if model_name == "RandomForest":
            display_name = "Random Forest"
        elif model_name == "LogisticRegression":
            display_name = "Logistic Regression"
        out["models"].append(display_name)
        out["f1_scores"].append(mm.get("f1_score"))   # None if absent
        out["f2_scores"].append(mm.get("f2_score"))
        out["precision"].append(mm.get("precision"))
        out["recall"].append(mm.get("recall"))
        out["auc_pr"].append(mm.get("auc_pr"))
        out["roc_auc"].append(mm.get("roc_auc"))
    
    return out

def calculate_roi_metrics(
    df,
    model_metrics: dict,
    model_name: str = "RandomForest",
    fraud_cost_multiplier: float = 4.41,
    recovery_rate: float = 1.0,
):
    """
    Base ROI model aligned with financial report.
    Uses 4.41x multiplier (includes OpEx, legal, investigation).
    """

    if df is None or 'Amount' not in df.columns or 'Class' not in df.columns:
        return {}

    models = model_metrics.get("models", [])
    if model_name not in models:
        if not models:
            return {}
        model_name = models[0]

    if model_name is None:
        return {}

    idx = models.index(model_name)
    precision = max(float(model_metrics["precision"][idx]), 1e-9)
    recall = max(min(float(model_metrics["recall"][idx]), 1.0), 0.0)

    fraud_df = df[df['Class'] == 1]
    total_fraud_amount = fraud_df['Amount'].sum()

    detected_amount = total_fraud_amount * recall
    recovered_value = detected_amount * fraud_cost_multiplier * recovery_rate

    undetected_loss = total_fraud_amount * (1 - recall) * fraud_cost_multiplier

    return {
        "model_used": model_name,
        "precision": precision,
        "recall": recall,
        "total_fraud_amount": float(total_fraud_amount),
        "recovered_economic_value": float(recovered_value),
        "undetected_fraud_cost": float(undetected_loss),
        "net_benefit": float(recovered_value - undetected_loss),
    }


def calculate_roi_metrics_auto(
    df,
    dataset_key=None,
    model_name="RandomForest",
):
    """
    Automatic ROI calculation aligned with the financial report.

    Assumptions:
    - Fraud cost multiplier (4.41x) already includes OpEx, investigation,
      legal and recovery costs.
    - Full recovery of detected fraud (base case).
    """

    model_metrics = get_model_metrics(dataset_key=dataset_key)
    if not model_metrics:
        return {}

    return calculate_roi_metrics(
        df=df,
        model_metrics=model_metrics,
        model_name=model_name,
        fraud_cost_multiplier=4.41,  # standard LexisNexis
        recovery_rate=1.0            # base case from the report
    )

