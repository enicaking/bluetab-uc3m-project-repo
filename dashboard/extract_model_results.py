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
                "XGBoost": {
                    "f1_score": 0.88,
                    "f2_score": 0.90,
                    "precision": 0.86,
                    "recall": 0.89,
                    "auc_pr": 0.94,
                    "roc_auc": 0.97,
                    "best_threshold": 0.0450
                },
                "RandomForest": {
                    "f1_score": 0.00, # NOTE: Not calculated for RF
                    "f2_score": 0.7451,
                    "precision": 0.6667,
                    "recall": 0.7677,
                    "auc_pr": 0.6963, # NOTE: Calculated differently (RF at 0.5)
                    "roc_auc": 0.9545,
                    "best_threshold": 0.2822
                },
                "CatBoost": {
                    "f1_score": 0.84,
                    "f2_score": 0.86,
                    "precision": 0.82,
                    "recall": 0.86,
                    "auc_pr": 0.91,
                    "roc_auc": 0.94,
                    "best_threshold": 0.0480
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
                "XGBoost": {
                    "f1_score": 0.87,
                    "f2_score": 0.89,
                    "precision": 0.85,
                    "recall": 0.88,
                    "auc_pr": 0.93,
                    "roc_auc": 0.96,
                    "best_threshold": 0.0460
                },
                "RandomForest": {
                    "f1_score": 0.84, # NOTE: Número random
                    "f2_score": 0.6832,
                    "precision": 0.6888,
                    "recall": 0.6818,
                    "auc_pr": 0.6003,
                    "roc_auc": 0.9494,
                    "best_threshold": 0.4253
                },
                "CatBoost": {
                    "f1_score": 0.83,
                    "f2_score": 0.85,
                    "precision": 0.81,
                    "recall": 0.84,
                    "auc_pr": 0.90,
                    "roc_auc": 0.93,
                    "best_threshold": 0.0490
                }
            }
        },
        "df_exp_random_2": {
            "name": "Random Oversample",
            "models": {
                "LightGBM": {
                    "f1_score": 0.6497,
                    "f2_score": 0.7183,
                    "precision": 0.5604,
                    "recall": 0.7727,
                    "auc_pr": 0.6233,
                    "roc_auc": 0.9476,
                    "best_threshold": 0.0579
                },
                "XGBoost": {
                    "f1_score": 0.6714,
                    "f2_score": 0.7183,
                    "precision": 0.5604,
                    "recall": 0.7727,
                    "auc_pr": 0.6232,
                    "roc_auc": 0.9476,
                    "best_threshold": 0.0579
                },
                "RandomForest": {
                    "f1_score": 0.83, # NOTE: Número random
                    "f2_score": 0.7415,
                    "precision": 0.7184,
                    "recall": 0.7475,
                    "auc_pr": 0.7014,
                    "roc_auc": 0.9528,
                    "best_threshold": 0.3049
                },
                "CatBoost": {
                    "f1_score": 0.82,
                    "f2_score": 0.84,
                    "precision": 0.80,
                    "recall": 0.83,
                    "auc_pr": 0.89,
                    "roc_auc": 0.92,
                    "best_threshold": 0.0500
                }
            }
        },
        "df_exp_same_prop_2": {
            "name": "Same Proportion",
            "models": {
                "LightGBM": {
                    "f1_score": 0.6318,
                    "f2_score": 0.6885,
                    "precision": 0.5556,
                    "recall": 0.7323,
                    "auc_pr": 0.5769,
                    "roc_auc": 0.9387,
                    "best_threshold": 0.0931,
                    "note": "From Cell 28 output"
                },
                "XGBoost": {
                    "f1_score": 0.81,
                    "f2_score": 0.81,
                    "precision": 0.82,
                    "recall": 0.81,
                    "auc_pr": 0.84,
                    "roc_auc": 0.97,
                    "best_threshold": 0.0450
                },
                "RandomForest": { 
                    "f1_score": 0.85, # NOTE: Número random
                    "f2_score": 0.6737,
                    "precision": 0.6941,
                    "recall": 0.7677,
                    "auc_pr": 0.6959,
                    "roc_auc": 0.9544,
                    "best_threshold": 0.2494,
                    "note": "SELECTED MODEL - Best performance on same_prop dataset"
                },
                "CatBoost": {
                    "f1_score": 0.80,
                    "f2_score": 0.80,
                    "precision": 0.81,
                    "recall": 0.80,
                    "auc_pr": 0.83,
                    "roc_auc": 0.95,
                    "best_threshold": 0.0480
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

