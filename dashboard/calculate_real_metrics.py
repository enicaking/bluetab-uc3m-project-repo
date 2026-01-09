"""
Script para calcular y guardar curvas ROC/PR reales y confusion matrices
desde los modelos entrenados guardados en archivos .pkl

BASADO EN: notebooks/modelos.ipynb

CÓMO EJECUTAR:
1. Asegúrate de que los modelos .pkl estén guardados en content/models/
2. Los modelos deben ser para df_same_prop:
   - RandomForest: rf_model_df_same_prop_new.pkl
   - CatBoost: catboost_model_df_same_prop_new.pkl
3. Ejecuta desde la raíz del proyecto:
   python dashboard/calculate_real_metrics.py

UBICACIÓN DE MODELOS:
El script SOLO busca los modelos en: content/models/

DATASET:
El script SOLO procesa el dataset df_exp_same_prop (el mejor dataframe)

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

def prepare_test_data(df, model, date_col='timestamp', class_col='Class'):
    """
    Prepare test data using temporal split (Day 2 = test set)
    Based on modelos.ipynb methodology: Day 1 = train, Day 2 = test
    Uses the exact features that the model expects
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
    
    # Get the exact features the model expects
    # Handle different model types: sklearn models, pipelines, etc.
    if hasattr(model, 'feature_names_'):
        expected_features = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
        # For RandomizedSearchCV or similar
        expected_features = model.best_estimator_.feature_names_in_
    elif hasattr(model, 'named_steps') or (hasattr(model, 'steps') and len(model.steps) > 0):
        # For Pipeline objects - try to get feature names from preprocessor
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # Try to get feature names from the preprocessor step
            preprocessor = None
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor', None)
            elif hasattr(model, 'steps'):
                for name, step in model.steps:
                    if 'preprocessor' in name.lower() or 'transform' in name.lower():
                        preprocessor = step
                        break
            
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                # We need the input features to get output features
                # For now, use a workaround: get from the classifier if it has feature_names_in_
                classifier = model.steps[-1][1] if hasattr(model, 'steps') else None
                if classifier and hasattr(classifier, 'feature_names_in_'):
                    expected_features = classifier.feature_names_in_
                else:
                    raise ValueError("Pipeline detected but cannot determine feature names. The model may need to be saved with feature names.")
            else:
                raise ValueError("Pipeline detected but preprocessor not found or doesn't have get_feature_names_out")
        else:
            raise ValueError("Cannot determine model features. Model must have feature_names_ or feature_names_in_")
    else:
        raise ValueError("Cannot determine model features. Model must have feature_names_ or feature_names_in_")
    
    print(f"    Model expects {len(expected_features)} features")
    
    # Prepare test dataframe
    test_df = df.loc[test_mask].copy()
    
    # Create amount_log if needed and not present
    if 'amount_log' in expected_features and 'amount_log' not in test_df.columns:
        if 'amount' in test_df.columns:
            test_df['amount_log'] = np.log1p(test_df['amount'])
        else:
            raise ValueError("Model expects 'amount_log' but 'amount' column not found in dataset")
    
    # Select only the features the model expects
    missing_features = [f for f in expected_features if f not in test_df.columns]
    if missing_features:
        print(f"    Warning: Missing features: {missing_features[:10]}... (total: {len(missing_features)})")
        # Try to handle one-hot encoded features (e.g., customer_country_Algeria)
        # Use pd.concat to avoid DataFrame fragmentation warnings
        missing_data = {feat: 0 for feat in missing_features}
        missing_df = pd.DataFrame(missing_data, index=test_df.index)
        test_df = pd.concat([test_df, missing_df], axis=1)
    
    # Ensure all expected features are present and in correct order
    X_test = test_df[expected_features].copy()
    y_test = test_df[class_col].copy()
    
    # Convert categorical columns to string for CatBoost compatibility
    # Identify columns that look categorical (object type or customer_country)
    categorical_like = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_like:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
    
    return X_test, y_test, expected_features

def update_model_results_with_real_curves():
    """Load models and calculate real curves, then update JSON"""
    
    # Load current JSON
    json_path = Path(__file__).parent / "data" / "model_results.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    # Paths
    models_base_dir = Path(__file__).parent.parent
    data_dir = models_base_dir / "content"
    models_dir = models_base_dir / "content" / "models"
    
    # Model files mapping - ONLY models in content/models for df_same_prop
    # Models are saved with: {model_type}_model_{dataset_name}.pkl
    model_files = {
        "RandomForest": {
            "df_exp_same_prop": "rf_model_df_same_prop_new.pkl",
        },
        "CatBoost": {
            "df_exp_same_prop": "catboost_model_df_same_prop_new.pkl",
        },
        "LogisticRegression": {
            "df_exp_same_prop": "lr_model_df_same_prop_new.pkl",
        }
    }
    
    # ONLY process df_exp_same_prop (the best dataset)
    target_dataset = "df_exp_same_prop"
    
    if target_dataset not in results_data["datasets"]:
        print(f"Error: Dataset '{target_dataset}' not found in model_results.json")
        return
    
    # Process only the target dataset
    dataset_key = target_dataset
    dataset_info = results_data["datasets"][dataset_key]
    print(f"\nProcessing dataset: {dataset_key}")
    
    # Load dataset
    csv_file = data_dir / f"{dataset_key}.csv"
    if not csv_file.exists():
        print(f"  Warning: CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")
    
    # Process each model
    for model_name, model_metrics in dataset_info["models"].items():
        print(f"\n  Processing model: {model_name}")
        
        # Check if model file exists
        model_file_map = model_files.get(model_name, {})
        model_file_name = model_file_map.get(dataset_key)
        
        if not model_file_name:
            print(f"    No model file mapping found")
            continue
        
        # ONLY look in content/models directory
        model_path = models_dir / model_file_name
        
        if not model_path.exists():
            print(f"    Model file not found: {model_path}")
            continue
        
        try:
            # Load model first to get expected features
            model = joblib.load(model_path)
            print(f"    Loaded model from: {model_path}")
            
            # Prepare test data using temporal split (Day 2 = test)
            # This uses the exact features the model expects
            try:
                X_test, y_test, expected_features = prepare_test_data(df, model, date_col='timestamp')
                print(f"    Test set: {len(X_test)} samples ({y_test.sum()} fraud cases, {y_test.sum()/len(y_test)*100:.2f}%)")
            except Exception as e:
                print(f"    Error preparing test data: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Handle CatBoost categorical features
            if model_name == "CatBoost":
                # CatBoost models store categorical feature indices
                # We need to ensure those features are strings
                if hasattr(model, 'get_cat_feature_indices'):
                    cat_indices = model.get_cat_feature_indices()
                    if cat_indices:
                        cat_feature_names = [expected_features[i] for i in cat_indices if i < len(expected_features)]
                        for col in cat_feature_names:
                            if col in X_test.columns:
                                # Convert to string and handle NaN
                                X_test[col] = X_test[col].astype(str).replace('nan', 'MISSING').fillna('MISSING')
                # Also handle object/category columns
                cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in cat_cols:
                    if col in X_test.columns:
                        X_test[col] = X_test[col].astype(str).replace('nan', 'MISSING').fillna('MISSING')
            
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
    print("1. Load models from content/models/ (ONLY df_same_prop models)")
    print("2. Load test data from content/ for df_exp_same_prop (Day 2 = test set)")
    print("3. Calculate real ROC/PR curves and confusion matrices")
    print("4. Update dashboard/data/model_results.json with real data")
    print("\n" + "=" * 80)
    print()
    
    update_model_results_with_real_curves()
    
    print("\n" + "=" * 80)
    print("Done! The dashboard will now use real curves and confusion matrices.")
    print("Only models from content/models/ for df_exp_same_prop were processed.")
    print("=" * 80)
