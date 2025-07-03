import pandas as pd
import numpy as np

# Set display options for better viewing in the console
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# --- 1. Load the Data ---
# Please update the file path to point to your 'train.csv' file.
try:
    df = pd.read_csv('train.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'train.csv' not found. Please update the file path.")
    # As a fallback for the rest of the script, create an empty DataFrame
    df = pd.DataFrame()

if not df.empty:
    # --- 2. Initial Inspection ---
    print("\n--- Initial Data Overview ---")
    print(f"Shape of the dataset (rows, columns): {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # --- 3. Missing Values Analysis ---
    print("\n--- Missing Values Analysis ---")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df) * 100).round(2)
    
    missing_summary = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    
    # Filter to show only columns with missing values and sort them
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )
    
    if missing_summary.empty:
        print("✅ No missing values found in the dataset.")
    else:
        print("Columns with missing values:")
        print(missing_summary)

    # --- 4. Identify Feature Types ---
    print("\n--- Feature Type Identification ---")
    
    # Exclude the target variable 'SalePrice' and the 'Id' column for this separation
    features = df.drop(columns=['SalePrice', 'Id'])
    
    # Identify numerical columns (we include int64 and float64)
    numerical_features = features.select_dtypes(include=np.number).columns.tolist()
    
    # Identify categorical columns (we select 'object' type)
    categorical_features = features.select_dtypes(include='object').columns.tolist()
    
    print(f"\nIdentified {len(numerical_features)} numerical features:")
    print(numerical_features)
    
    print(f"\nIdentified {len(categorical_features)} categorical features:")
    print(categorical_features)
