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
CSV_EXPORTS = Path(__file__).parent.parent / "notebooks" / "csv_exports"

def load_data(dataset_name="df_exp_50_2.csv"):
    """Load dataset from csv_exports"""
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

def get_temporal_data(df, freq='H'):
    """Get temporal aggregation of transactions"""
    if df is None or 'timestamp' not in df.columns:
        return None
    # Analysis by hour
    df_temp = df.copy()
    df_temp['hour'] = df_temp['timestamp'].dt.floor(freq)
    # Group by hour and calculate the statistics
    temporal = df_temp.groupby('hour').agg({
        'Class': ['count', 'sum', 'mean'],      # count: number of transactions, sum: number of fraudulent transactions, mean:percentage of fraudulent transactions (0-1)
        'Amount': ['sum', 'mean']               # sum: total amount of transactions, mean: average amount of transactions
    }).reset_index()
    
    temporal.columns = ['timestamp', 'count', 'fraud_count', 'fraud_rate', 'total_amount', 'avg_amount']
    temporal['fraud_rate'] = temporal['fraud_rate'] * 100
    return temporal

def get_country_stats(df, min_tx=100, m=500):
    """
    Country stats for dashboard.
    - min_tx: minimum number of transactions to appear in the ranking
    - m: smoothing force (higher => more 'pull' towards the global mean)
    """
    if df is None or 'customer_country' not in df.columns or 'Class' not in df.columns:
        return None

    cs = df.groupby('customer_country').agg(
        total=('Class', 'count'),
        fraud_count=('Class', 'sum'),
        total_amount=('Amount', 'sum') if 'Amount' in df.columns else ('Class', 'count')
    ).reset_index().rename(columns={'customer_country': 'country'})

    # global rate (0-1)
    global_rate = df['Class'].mean()

    # raw rate (0-1)
    cs['fraud_rate'] = cs['fraud_count'] / cs['total']

    # smoothed rate (0-1)
    cs['fraud_rate_smoothed'] = (cs['fraud_count'] + m * global_rate) / (cs['total'] + m)

    # to %
    cs['fraud_rate_pct'] = cs['fraud_rate'] * 100
    cs['fraud_rate_smoothed_pct'] = cs['fraud_rate_smoothed'] * 100

    # filter by volume
    cs = cs[cs['total'] >= min_tx].copy()

    # sort by the smoothed rate
    return cs.sort_values('fraud_rate_smoothed_pct', ascending=False)


def get_model_metrics(dataset_key=None):
    """
    Get model performance metrics from saved JSON results.
    
    Args:
        dataset_key: Optional dataset key (e.g., 'df_exp_same_prop_2'). 
                    If None, uses the best model dataset.
    
    Returns:
        dict with model metrics for all models in the specified dataset
    """
    import json
    
    # Path to model results JSON
    json_path = Path(__file__).parent / "data" / "model_results.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Model results file not found at {json_path}. Using fallback data.")
        return {
            'models': ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest'],
            'f1_scores': [0.89, 0.91, 0.88, 0.85],
            'f2_scores': [0.92, 0.94, 0.90, 0.87],
            'precision': [0.87, 0.89, 0.86, 0.83],
            'recall': [0.91, 0.93, 0.89, 0.87],
            'auc_pr': [0.95, 0.97, 0.94, 0.92],
            'roc_auc': [0.98, 0.99, 0.97, 0.95],
        }
    except Exception as e:
        print(f"Error loading model results: {e}. Using fallback data.")
        return {
            'models': ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest'],
            'f1_scores': [0.89, 0.91, 0.88, 0.85],
            'f2_scores': [0.92, 0.94, 0.90, 0.87],
            'precision': [0.87, 0.89, 0.86, 0.83],
            'recall': [0.91, 0.93, 0.89, 0.87],
            'auc_pr': [0.95, 0.97, 0.94, 0.92],
            'roc_auc': [0.98, 0.99, 0.97, 0.95],
        }
    
    # Use specified dataset or default to best model's dataset
    if dataset_key is None:
        dataset_key = results_data.get('best_model', {}).get('dataset', 'df_exp_same_prop_2')
    
    # Get dataset info
    dataset_info = results_data.get('datasets', {}).get(dataset_key, {})
    
    if not dataset_info:
        # Fallback to best model if dataset not found
        best_model = results_data.get('best_model', {})
        return {
            'models': [best_model.get('model', 'RandomForest')],
            'f1_scores': [best_model.get('f1_score', 0.85)],
            'f2_scores': [best_model.get('f2_score', 0)],
            'precision': [best_model.get('precision', 0)],
            'recall': [best_model.get('recall', 0)],
            'auc_pr': [best_model.get('auc_pr', 0)],
            'roc_auc': [best_model.get('roc_auc', 0)],
        }
    
    # Extract metrics for all models in the dataset
    models = []
    f1_scores = []  # TODO: no hay f1 score en el notebook. quitarlo
    f2_scores = []
    precision_scores = []
    recall_scores = []
    auc_pr_scores = []
    roc_auc_scores = []
    
    models_dict = dataset_info.get('models', {})
    
    # Order: LightGBM, XGBoost, CatBoost, RandomForest (to match original order)
    model_order = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest']
    
    for model_name in model_order:
        if model_name in models_dict:
            model_metrics = models_dict[model_name]
            models.append(model_name if model_name != 'RandomForest' else 'Random Forest')
            f1_scores.append(model_metrics.get('f1_score', 0.0))
            f2_scores.append(model_metrics.get('f2_score', 0.0))
            precision_scores.append(model_metrics.get('precision', 0.0))
            recall_scores.append(model_metrics.get('recall', 0.0))
            auc_pr_scores.append(model_metrics.get('auc_pr', 0.0))
            roc_auc_scores.append(model_metrics.get('roc_auc', 0.0))
    
    return {
        'models': models,
        'f1_scores': f1_scores,
        'f2_scores': f2_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'auc_pr': auc_pr_scores,
        'roc_auc': roc_auc_scores,
    }


def calculate_roi_metrics(  # TODO: se puede incluir predicciones
    df,
    model_metrics: dict,
    model_name: str = "LightGBM",
    cost_per_investigation: float = 50,
    recovery_rate: float = 1.0,
):
    """
    ROI estimation using real model precision/recall from saved results.
    """

    if df is None or 'Amount' not in df.columns or 'Class' not in df.columns:
        return {}

    # 1) get precision/recall for chosen model
    models = model_metrics.get("models", [])
    if model_name not in models:
        if not models:
            return {}
        model_name = models[0]

    idx = models.index(model_name)

    # usa .get para evitar KeyError si faltan keys
    precision_list = model_metrics.get("precision", [])
    recall_list = model_metrics.get("recall", [])
    if idx >= len(precision_list) or idx >= len(recall_list):
        return {}

    precision = float(precision_list[idx])
    recall = float(recall_list[idx])

    # safety
    precision = max(precision, 1e-9)
    recall = max(min(recall, 1.0), 0.0)

    # 2) fraud totals from data
    fraud_df = df[df['Class'] == 1]
    total_fraud_amount = float(fraud_df['Amount'].sum())
    total_frauds = int(len(fraud_df))

    # 3) expected detected frauds and recovered amount
    detected_frauds = total_frauds * recall
    recovered_amount = total_fraud_amount * recall * recovery_rate

    # 4) alerts and false positives
    total_alerts = detected_frauds / precision
    false_positives = total_alerts - detected_frauds

    investigation_cost = total_alerts * cost_per_investigation
    net_benefit = recovered_amount - investigation_cost

    return {
        "model_used": model_name,
        "precision": precision,
        "recall": recall,
        "total_fraud_amount": total_fraud_amount,
        "fraud_detected_pct": recall * 100,
        "alerts_generated": int(round(total_alerts)),
        "false_positives": int(round(false_positives)),
        "investigation_cost": float(investigation_cost),
        "recovered_amount": float(recovered_amount),
        "net_benefit": float(net_benefit),
    }


def calculate_roi_metrics_auto(
    df,
    dataset_key=None,
    model_name="LightGBM",
    cost_per_investigation=50,
    recovery_rate=1.0
):
    model_metrics = get_model_metrics(dataset_key=dataset_key)
    if not model_metrics:
        return {}

    return calculate_roi_metrics(
        df=df,
        model_metrics=model_metrics,
        model_name=model_name,
        cost_per_investigation=cost_per_investigation,
        recovery_rate=recovery_rate
    )
