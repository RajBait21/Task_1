# ===============================
# TASK 1: DATA PIPELINE DEVELOPMENT
# ===============================

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------------
# 1. EXTRACT (Load Data)
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully")
    return df

# -------------------------------
# 2. TRANSFORM (Preprocessing)
# -------------------------------
def preprocess_data(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    transformed_data = preprocessor.fit_transform(df)
    print("Data Preprocessing Completed")
    return transformed_data

# -------------------------------
# 3. LOAD (Save Processed Data)
# -------------------------------
def save_data(transformed_data, output_file):
    pd.DataFrame(transformed_data.toarray() if hasattr(transformed_data, "toarray") else transformed_data)\
        .to_csv(output_file, index=False)
    print("Processed Data Saved Successfully")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    input_file = "data.csv"           # Input dataset
    output_file = "processed_data.csv"  # Output dataset

    df = load_data(input_file)
    transformed_data = preprocess_data(df)
    save_data(transformed_data, output_file)
