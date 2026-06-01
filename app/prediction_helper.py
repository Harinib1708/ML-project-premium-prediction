import pandas as pd
import numpy as np
import joblib

# Load models and scalers using Mac/Linux compatible forward slashes
model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # FIX: Convert the incoming string completely to lowercase, remove spaces around '&'
    normalized_history = medical_history.lower().replace(" & ", "&")
    diseases = normalized_history.split("&")

    # Calculate the total risk score by stripping whitespace from entries
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)

    max_score = 14  # risk score for heart disease (8) + second max risk score (6)
    min_score = 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def preprocess_input(input_dict):
    # Define the expected columns
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Initialize a 2D DataFrame with zeros using NumPy to satisfy IDE type-checkers
    df = pd.DataFrame(np.zeros((1, len(expected_columns))), columns=expected_columns, index=[0])

    # Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Calculate normalized risk score using medical history safely
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Scale numerical values
    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    # Select the proper scaler based on age group
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # FIX: Supply 0.0 instead of None to prevent structural scaling type crashes
    df['income_level'] = 0.0

    # Apply transformation on specific tracking columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Drop the temporary column
    df.drop('income_level', axis='columns', inplace=True)

    return df


def predict(input_dict):
    # 1. Print the raw dictionary payload from Streamlit to the terminal console
    print("\n--- Raw Input Received from UI ---")
    print(input_dict)

    # 2. Preprocess the dictionary into the clean 2D DataFrame table
    input_df = preprocess_input(input_dict)

    # 3. Print the formatted structure to visually verify encoding transformations
    print("\n--- Processed 2D DataFrame sent to Model ---")
    print(input_df.to_string())
    print("-----------------------------------------\n")

    # 4. Use the 2D DataFrame layout to run predictions
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    # 5. FIX: Print out the exact text "predicted class" followed by the value
    predicted_value = int(prediction[0])
    print(f"predicted class: {predicted_value}")
    print("=========================================\n")

    return predicted_value