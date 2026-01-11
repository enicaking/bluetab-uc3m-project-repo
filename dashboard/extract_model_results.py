"""
Script para extraer resultados de modelos desde modeling3.ipynb
y guardarlos en formato JSON/Excel para el dashboard
"""
import pandas as pd
import json
from pathlib import Path

# Basado en los resultados del notebook modeling3.ipynb
# El usuario mencionó que finalmente seleccionaron Random Forest con same proportions

# Estructura de resultados esperada
# NOTE: No se hizo XGBoost, y el CatBoost solo para el 'mejor' modelo. Hay que cambiar este json y volver a generar resultados.
results_data = {
    "datasets": {
        "df_exp_50_2": { 
            "name": "Balanced 50/50",
            "models": {
                "LightGBM": {
                    "f1_score": 0.6890,  
                    "f2_score": 0.7115,
                    "precision": 0.6545,
                    "recall": 0.7273,
                    "auc_pr": 0.6442,
                    "roc_auc": 0.9486,
                    "best_threshold": 0.7796
                },
                "XGBoost": { # eval_desbalanceado_xgboost
                    "f1_score": 0.6884,
                    "f2_score": 0.6905,
                    "precision": 0.6850,
                    "recall": 0.6919,
                    "auc_pr": 0.6253,
                    "roc_auc": 0.5488,
                    "best_threshold": 0.9497
                },
                "RandomForest": {
                    "f1_score": 0.7136, 
                    "f2_score": 0.7451,
                    "precision": 0.6667,
                    "recall": 0.7677,
                    "auc_pr": 0.6963, 
                    "roc_auc": 0.9545,
                    "best_threshold": 0.2822
                },
                "CatBoost": { # eval_desbalanceado_catboost
                    "f1_score": 0.6974,
                    "f2_score": 0.6911,
                    "precision": 0.7083,
                    "recall": 0.6869,
                    "auc_pr": 0.6563,
                    "roc_auc": 0.9164,
                    "best_threshold": 0.9311
                }
            }
        },
        "df_exp_63_2": {
            "name": "Balanced 63/37",
            "models": {
                "LightGBM": {
                    "f1_score": 0.6288,
                    "f2_score": 0.6539,
                    "precision": 0.5911,
                    "recall": 0.6717,
                    "auc_pr": 0.5626,
                    "roc_auc": 0.9289,
                    "best_threshold": 0.8704
                },
                "XGBoost": { # eval_desbalanceado_xgboost
                    "f1_score": 0.6465,
                    "f2_score": 0.6465,
                    "precision": 0.6465,
                    "recall": 0.6465,
                    "auc_pr": 0.5373,
                    "roc_auc": 0.9331,
                    "best_threshold": 0.9704
                },
                "RandomForest": {
                    "f1_score": 0.6853, 
                    "f2_score": 0.6832,
                    "precision": 0.6888,
                    "recall": 0.6818,
                    "auc_pr": 0.6003,
                    "roc_auc": 0.9494,
                    "best_threshold": 0.4253
                },
                "CatBoost": { # eval_desbalanceado_catboost
                    "f1_score": 0.6829,
                    "f2_score": 0.6542,
                    "precision": 0.7368,
                    "recall": 0.6364,
                    "auc_pr": 0.5875,
                    "roc_auc": 0.9314,
                    "best_threshold": 0.9775
                }
            }
        },
        "df_exp_random_2": {
            "name": "Random Oversample",
            "models": {
                "LightGBM": { # eval_desbalanceado_lgbm_all_vars
                    "f1_score": 0.7976,
                    "f2_score": 0.7803,
                    "precision": 0.8281,
                    "recall": 0.7692,
                    "auc_pr": 0.7886,
                    "roc_auc": 0.9607,
                    "best_threshold": 0.0333
                },
                "XGBoost": { # eval_desbalanceado_xgboost_all_vars
                    "f1_score": 0.7830,
                    "f2_score": 0.7501,
                    "precision": 0.8448,
                    "recall": 0.7296,
                    "auc_pr": 0.7624,
                    "roc_auc": 0.9673,
                    "best_threshold": 0.2654
                },
                "RandomForest": { # eval_desbalanceado_rf_all_vars
                    "f1_score": 0.8035, 
                    "f2_score": 0.7767,
                    "precision": 0.8523,
                    "recall": 0.7599,
                    "auc_pr": 0.7831,
                    "roc_auc": 0.9634,
                    "best_threshold": 0.3766
                },
                "CatBoost": { # eval_desbalanceado_catboost_all_vars
                    "f1_score": 0.7886,
                    "f2_score": 0.7690,
                    "precision": 0.8236,
                    "recall": 0.7564,
                    "auc_pr": 0.7689,
                    "roc_auc": 0.9343,
                    "best_threshold": 0.6025
                }
            }
        },
        "df_exp_same_prop_2": {
            "name": "Same Proportion",
            "models": {
                "LightGBM": { # eval_desbalanceado_lgbm_all_vars
                    "f1_score": 0.8220,
                    "f2_score": 0.8034,
                    "precision": 0.8552,
                    "recall": 0.7914,
                    "auc_pr": 0.8012,
                    "roc_auc": 0.9600,
                    "best_threshold": 0.0203,
                    "note": "From Cell 28 output"
                },
                "XGBoost": { # eval_desbalanceado_xgboost_all_vars
                    "f1_score": 0.7717,
                    "f2_score": 0.7589,
                    "precision": 0.7941,
                    "recall": 0.7506,
                    "auc_pr": 0.7683,
                    "roc_auc": 0.9692,
                    "best_threshold": 0.1957
                },
                "RandomForest": { # eval_desbalanceado_rf_all_vars
                    "f1_score": 0.8125, # NOTE: hyperparametrization does not improve
                    "f2_score": 0.7867,
                    "precision": 0.8596,
                    "recall": 0.7704,
                    "auc_pr": 0.7971,
                    "roc_auc": 0.9678,
                    "best_threshold": 0.3832,
                    "note": "SELECTED MODEL - Best performance on same_prop dataset"
                },
                "CatBoost": { # eval_desbalanceado_catboost_all_vars
                    "f1_score": 0.8179,
                    "f2_score": 0.7799,
                    "precision": 0.8903,
                    "recall": 0.7564,
                    "auc_pr": 0.7746,
                    "roc_auc": 0.9272,
                    "best_threshold": 0.6827
                }
            }
        }
    },
    "best_model": {
        "dataset": "df_exp_same_prop_2",
        "model": "RandomForest",
        "f1_score": 0, # TODO: no hay f1 score en el notebook
        "f2_score": 0.8329,
        "precision": 0.9149,
        "recall": 0.8147,
        "auc_pr": 0.8481,
        "roc_auc": 0.9835,
        "best_threshold": 0.3434
    },
    "metadata": {
        "extraction_date": "2026-01-07",
        "source": "modelos.ipynb",
        "note": "Random Forest with same_prop dataset was selected as final model"
    }
}

# Guardar como JSON
output_dir = Path(__file__).parent.parent / "dashboard" / "data"
output_dir.mkdir(exist_ok=True)

json_path = output_dir / "model_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print(f"JSON guardado en: {json_path}")

# Crear DataFrame para Excel
rows = []
for dataset_key, dataset_info in results_data["datasets"].items():
    for model_name, model_metrics in dataset_info["models"].items():
        rows.append({
            "dataset": dataset_key,
            "dataset_name": dataset_info["name"],
            "model": model_name,
            "f1_score": model_metrics["f1_score"],
            "f2_score": model_metrics["f2_score"],
            "precision": model_metrics["precision"],
            "recall": model_metrics["recall"],
            "auc_pr": model_metrics["auc_pr"],
            "roc_auc": model_metrics["roc_auc"],
            "best_threshold": model_metrics["best_threshold"]
        })

df_results = pd.DataFrame(rows)

# Guardar como Excel
excel_path = output_dir / "model_results.xlsx"
try:
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All Results', index=False)
        
        # Crear hoja con el mejor modelo
        best_df = pd.DataFrame([results_data["best_model"]])
        best_df.to_excel(writer, sheet_name='Best Model', index=False)
        
        # Crear hoja resumen por dataset
        summary = df_results.groupby('dataset').agg({
            'f2_score': 'max',
            'f1_score': 'max',
            'roc_auc': 'max'
        }).reset_index()
        summary.to_excel(writer, sheet_name='Summary by Dataset', index=False)
    print(f"Excel guardado en: {excel_path}")
except ImportError:
    print("openpyxl no instalado, guardando solo JSON")
    # Guardar como CSV como alternativa
    csv_path = output_dir / "model_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"CSV guardado en: {csv_path}")

print(f"\nTotal de resultados: {len(df_results)}")
print(f"\nMejor modelo: {results_data['best_model']['model']} en {results_data['best_model']['dataset']}")
print(f"  F2 Score: {results_data['best_model']['f2_score']:.4f}")
print(f"  ROC-AUC: {results_data['best_model']['roc_auc']:.4f}")

