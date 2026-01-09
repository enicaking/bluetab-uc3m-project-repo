"""
Script para calcular y guardar curvas ROC/PR reales y confusion matrices
desde los modelos entrenados guardados en archivos .pkl

BASADO EN: notebooks/modelos.ipynb

CÓMO EJECUTAR:
1. Asegúrate de que los modelos .pkl estén guardados (se generan al ejecutar modelos.ipynb)
2. Los modelos se guardan con nombres como:
   - RandomForest: rf_model_{dataset_name}.pkl (ej: rf_model_df_same_prop_new.pkl)
   - LightGBM: lgb_model_{dataset_name}.pkl
   - CatBoost: catboost_model_{dataset_name}.pkl
   - XGBoost: xgb_model_{dataset_name}.pkl
3. Ejecuta desde la raíz del proyecto:
   python dashboard/calculate_real_metrics.py

UBICACIÓN DE MODELOS:
El script busca los modelos en:
- Raíz del proyecto
- notebooks/
- Ruta absoluta si se especifica

METODOLOGÍA:
- Usa split temporal: Día 1 (2023-01-01) = train, Día 2 (2023-01-02) = test
- Calcula curvas ROC/PR reales desde las predicciones del modelo
- Calcula confusion matrices en threshold óptimo (F2) y threshold 0.5
- Actualiza model_results.json con los datos reales
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score, fbeta_score
)
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def calculate_real_metrics_for_model(model, X_test, y_test, model_name="Model"):
    """Calculate real ROC/PR curves and confusion matrix from model predictions"""
    try:
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Calculate PR curve
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)
        
        # Calculate confusion matrix at threshold 0.5
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics at threshold 0.5
        precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_05 = f1_score(y_test, y_pred)
        f2_05 = fbeta_score(y_test, y_pred, beta=2)
        
        # Find best threshold for F2
        best_thresh = 0.5
        best_f2 = f2_05
        best_f1 = f1_05
        best_precision = precision_05
        best_recall = recall_05
        
        for thresh in np.arange(0.01, 0.99, 0.01):
            y_pred_th = (y_proba >= thresh).astype(int)
            if y_pred_th.sum() > 0:  # At least one positive prediction
                f2_th = fbeta_score(y_test, y_pred_th, beta=2)
                if f2_th > best_f2:
                    best_f2 = f2_th
                    best_thresh = thresh
                    best_f1 = f1_score(y_test, y_pred_th)
                    cm_th = confusion_matrix(y_test, y_pred_th)
                    if cm_th.shape == (2, 2):
                        tn_th, fp_th, fn_th, tp_th = cm_th.ravel()
                        best_precision = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 0.0
                        best_recall = tp_th / (tp_th + fn_th) if (tp_th + fn_th) > 0 else 0.0
        
        # Confusion matrix at best threshold
        y_pred_best = (y_proba >= best_thresh).astype(int)
        cm_best = confusion_matrix(y_test, y_pred_best)
        tn_best, fp_best, fn_best, tp_best = cm_best.ravel()
        
        return {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr],
            'roc_auc': float(roc_auc),
            'precision_arr': [float(x) for x in precision_arr],
            'recall_arr': [float(x) for x in recall_arr],
            'auc_pr': float(auc_pr),
            'confusion_matrix': {
                'at_05': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                'at_best_thresh': {'tn': int(tn_best), 'fp': int(fp_best), 'fn': int(fn_best), 'tp': int(tp_best)}
            },
            'f1_score': float(best_f1),
            'f2_score': float(best_f2),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'best_threshold': float(best_thresh)
        }
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        return None

def prepare_test_data(df, date_col='timestamp', class_col='Class', pca_cols=None):
    """
    Prepare test data using temporal split (Day 2 = test set)
    Based on modelos.ipynb methodology: Day 1 = train, Day 2 = test
    """
    # Try timestamp first, then date
    date_col_actual = None
    for col in ['timestamp', 'date']:
        if col in df.columns:
            date_col_actual = col
            break
    
    if date_col_actual:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col_actual]):
            df[date_col_actual] = pd.to_datetime(df[date_col_actual], errors='coerce')
        
        # Get unique dates (sorted)
        dates = sorted(df[date_col_actual].dt.date.unique())
        if len(dates) >= 2:
            train_date = dates[0]  # Day 1 = training
            test_date = dates[1]   # Day 2 = test
            test_mask = df[date_col_actual].dt.date == test_date
            print(f"    Temporal split: Train={train_date}, Test={test_date}")
        else:
            # Fallback to train_test_split if only one day
            from sklearn.model_selection import train_test_split
            test_mask = np.zeros(len(df), dtype=bool)
            _, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df[class_col] if class_col in df.columns else None)
            test_mask[test_idx] = True
            print(f"    Warning: Only one day found, using random split")
    else:
        # Fallback to train_test_split
        from sklearn.model_selection import train_test_split
        test_mask = np.zeros(len(df), dtype=bool)
        _, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df[class_col] if class_col in df.columns else None)
        test_mask[test_idx] = True
        print(f"    Warning: No timestamp/date column found, using random split")
    
    # Get PCA columns if not specified
    if pca_cols is None:
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
    
    X_test = df.loc[test_mask, pca_cols].copy()
    y_test = df.loc[test_mask, class_col].copy()
    
    return X_test, y_test

def update_model_results_with_real_curves():
    """Load models and calculate real curves, then update JSON"""
    
    # Load current JSON
    json_path = Path(__file__).parent / "data" / "model_results.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    # Paths
    models_base_dir = Path(__file__).parent.parent
    data_dir = models_base_dir / "content" 

    
    # Model files mapping - based on naming convention from modelos.ipynb
    # Models are saved with: {model_type}_model_{dataset_name}.pkl
    # NOTE: If models don't exist, the dashboard will use approximate curves from JSON metrics
    model_files = {
        "RandomForest": {
            "df_exp_same_prop": "rf_model_df_same_prop_new.pkl",  # From modelos.ipynb
            "df_exp_50": "rf_model_df_exp_50.pkl",
            "df_exp_63": "rf_model_df_exp_63.pkl",
            "df_exp_random": "rf_model_df_exp_random.pkl",
        },
        "LightGBM": {
            "df_exp_same_prop": "lgb_model_df_same_prop.pkl",
            "df_exp_50": "lgb_model_df_exp_50.pkl",
            "df_exp_63": "lgb_model_df_exp_63.pkl",
            "df_exp_random": "lgb_model_df_exp_random.pkl",
        },
        "XGBoost": {
            "df_exp_same_prop": "xgb_model_df_same_prop.pkl",
            "df_exp_50": "xgb_model_df_exp_50.pkl",
            "df_exp_63": "xgb_model_df_exp_63.pkl",
            "df_exp_random": "xgb_model_df_exp_random.pkl",
        },
        "CatBoost": {
            "df_exp_same_prop": "catboost_model_df_same_prop_new.pkl",  # From modelos.ipynb
            "df_exp_50": "catboost_model_df_exp_50.pkl",
            "df_exp_63": "catboost_model_df_exp_63.pkl",
            "df_exp_random": "catboost_model_df_exp_random.pkl",
        }
    }
    
    # Process each dataset and model
    for dataset_key, dataset_info in results_data["datasets"].items():
        print(f"\nProcessing dataset: {dataset_key}")
        
        # Load dataset
        csv_file = data_dir / f"{dataset_key}.csv"
        if not csv_file.exists():
            print(f"  Warning: CSV file not found: {csv_file}")
            continue
        
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} rows")
        
        # Get PCA columns
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        if len(pca_cols) == 0:
            print(f"  Warning: No PCA columns found")
            continue
        
        print(f"  Found {len(pca_cols)} PCA columns")
        
        # Prepare test data using temporal split (Day 2 = test)
        try:
            X_test, y_test = prepare_test_data(df, date_col='timestamp')
            print(f"  Test set: {len(X_test)} samples ({y_test.sum()} fraud cases, {y_test.sum()/len(y_test)*100:.2f}%)")
        except Exception as e:
            print(f"  Error preparing test data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Process each model
        for model_name, model_metrics in dataset_info["models"].items():
            print(f"\n  Processing model: {model_name}")
            
            # Check if model file exists
            model_file_map = model_files.get(model_name, {})
            model_file_name = model_file_map.get(dataset_key)
            
            if not model_file_name:
                print(f"    No model file mapping found")
                continue
            
            # Try multiple possible locations
            possible_paths = [
                models_base_dir / model_file_name,
                models_base_dir / "notebooks" / model_file_name,
                Path(model_file_name),
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if not model_path or not model_path.exists():
                print(f"    Model file not found: {model_file_name}")
                print(f"    Tried paths: {possible_paths}")
                continue
            
            try:
                # Load model
                model = joblib.load(model_path)
                print(f"    Loaded model from: {model_path}")
                
                # Calculate real metrics
                real_metrics = calculate_real_metrics_for_model(model, X_test, y_test, f"{model_name}_{dataset_key}")
                
                if real_metrics:
                    # Update results with real metrics
                    model_metrics.update(real_metrics)
                    print(f"    Updated with real metrics")
                    print(f"      F2 Score: {real_metrics['f2_score']:.4f}")
                    print(f"      ROC-AUC: {real_metrics['roc_auc']:.4f}")
                    print(f"      AUC-PR: {real_metrics['auc_pr']:.4f}")
                else:
                    print(f"    Failed to calculate metrics")
            except Exception as e:
                print(f"    Error loading/calculating for {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated JSON saved to: {json_path}")

if __name__ == "__main__":
    print("=" * 80)
    print("Calculating Real ROC/PR Curves and Confusion Matrices")
    print("Based on: notebooks/modelos.ipynb")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Load models from .pkl files (saved when running modelos.ipynb)")
    print("2. Load test data from content/ (Day 2 = test set)")
    print("3. Calculate real ROC/PR curves and confusion matrices")
    print("4. Update dashboard/data/model_results.json with real data")
    print("\n" + "=" * 80)
    print()
    
    update_model_results_with_real_curves()
    
    print("\n" + "=" * 80)
    print("Done! The dashboard will now use real curves and confusion matrices.")
    print("If models were not found, the dashboard will use approximate values")
    print("from model_results.json (extracted from notebook outputs).")
    print("=" * 80)
