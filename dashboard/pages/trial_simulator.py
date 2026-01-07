import joblib  
import pandas as pd
import numpy as np

catboost_model = joblib.load("dashboard/models/catboost_model_df_same_prop_new.pkl")
lr_model = joblib.load("dashboard/models/lr_model_df_same_prop_new.pkl")
rf_hyper_model = joblib.load("dashboard/models/rf_hyperparameter_model_df_same_prop_new.pkl")
rf_model = joblib.load("dashboard/models/rf_model_df_same_prop_new.pkl")

# print(catboost_model)
# print(catboost_model.feature_names_)

def simulate_model(model, customer_country, amount):
    # 1. Transform the amount to amount_log
    # We add 1 to avoid log(0) errors
    amount_log = np.log1p(amount)
    
    # 2. Define the "Average" state for the missing V variables
    # In a real app, these should be the actual medians from your training set
    neutral_v_value = 0.0 
    
    # 3. Create the full input vector (All 17 features)
    # Order must match your model.feature_names_ exactly
    input_data = {
        'V28': neutral_v_value, 'V19': neutral_v_value, 'V13': neutral_v_value,
        'V14': neutral_v_value, 'V24': neutral_v_value, 'V27': neutral_v_value,
        'V17': neutral_v_value, 'V26': neutral_v_value, 'V6': neutral_v_value,
        'amount_log': amount_log,
        'V23': neutral_v_value, 'V15': neutral_v_value,
        'customer_country': customer_country,
        'V20': neutral_v_value, 'V21': neutral_v_value,
        'V3': neutral_v_value, 'V25': neutral_v_value
    }
    
    # 4. Convert to DataFrame and Ensure correct order
    input_df = pd.DataFrame([input_data])[model.feature_names_]
    
    # 5. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    probability_percentage = float(probability) * 100
    formatted_probability = f"{probability_percentage:.5f}"
    
    return int(prediction), formatted_probability

results = simulate_model(catboost_model, "Russia", int(50000000000000))
print(results)

# print(type(catboost_model))
