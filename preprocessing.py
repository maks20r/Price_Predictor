import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import sys
from time import time

def log_progress(message):
    """Print timestamped progress message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def convert_dates(date_str):
    """
    Convert dates to datetime, handling both Gregorian and potential Hijri dates.
    Returns None for invalid dates.
    """
    try:
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    except:
        return pd.NaT

def preprocess_dubai_real_estate(df, output_file='preprocessed_real_estate.csv'):
    """
    Preprocess Dubai real estate data for temporal analysis and price prediction.
    Removes Arabic text and focuses on English fields only.
    
    Parameters:
    df (pandas.DataFrame): Raw real estate transaction data
    output_file (str): Path to save the preprocessed dataset
    
    Returns:
    pandas.DataFrame: Preprocessed data ready for analysis
    dict: Mapping dictionaries for encoded categorical variables
    """
    start_time = time()
    log_progress("Starting preprocessing pipeline...")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    log_progress(f"Initial dataset shape: {data.shape}")
    
    # Remove all Arabic columns
    arabic_columns = [col for col in data.columns if col.endswith('_ar')]
    data = data.drop(columns=arabic_columns)
    log_progress(f"Removed {len(arabic_columns)} Arabic columns")
    
    # Keep only essential transaction IDs
    data = data.drop(columns=['transaction_id', 'procedure_id', 'trans_group_id'])
    log_progress("Removed unnecessary ID columns")
    
    # Convert dates safely
    log_progress("Converting dates...")
    original_count = len(data)
    data['instance_date'] = data['instance_date'].apply(convert_dates)
    
    # Remove rows with invalid dates
    data = data.dropna(subset=['instance_date'])
    invalid_dates = original_count - len(data)
    log_progress(f"Removed {invalid_dates} rows with invalid dates")
    
    # Filter dates to a reasonable range
    date_mask = (data['instance_date'] >= '1990-01-01') & (data['instance_date'] <= '2024-12-31')
    data = data[date_mask]
    log_progress(f"Filtered to dates between 1990 and 2024. Remaining rows: {len(data)}")
    
    # Sort by date
    data = data.sort_values('instance_date')
    
    # Extract temporal features
    log_progress("Creating temporal features...")
    data['year'] = data['instance_date'].dt.year
    data['month'] = data['instance_date'].dt.month
    data['quarter'] = data['instance_date'].dt.quarter
    data['day_of_week'] = data['instance_date'].dt.dayofweek
    
    # Create price-related features
    log_progress("Processing price-related features...")
    data['price_per_sqm'] = data['actual_worth'] / data['procedure_area']
    
    # Handle missing values
    log_progress("Handling missing values...")
    numeric_columns = ['procedure_area', 'actual_worth', 'meter_sale_price', 
                      'rent_value', 'meter_rent_price']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        missing_before = data[col].isna().sum()
        data[col] = data.groupby('property_type_en')[col].transform(
            lambda x: x.fillna(x.median()))
        missing_after = data[col].isna().sum()
        log_progress(f"Filled {missing_before - missing_after} missing values in {col}")
    
    # Create binary flags
    log_progress("Creating property usage flags...")
    data['is_residential'] = data['property_usage_en'].str.contains('Residential', na=False).astype(int)
    data['is_commercial'] = data['property_usage_en'].str.contains('Commercial', na=False).astype(int)
    
    # Clean text columns
    log_progress("Cleaning text columns...")
    text_columns = ['property_type_en', 'property_usage_en', 'reg_type_en', 
                   'area_name_en', 'trans_group_en', 'procedure_name_en',
                   'nearest_landmark_en', 'nearest_metro_en', 'nearest_mall_en']
    
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown').str.strip()
    
    # Encode categorical variables
    log_progress("Encoding categorical variables...")
    categorical_columns = ['property_type_en', 'reg_type_en', 'area_name_en', 
                         'trans_group_en', 'procedure_name_en']
    encoders = {}
    
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            encoders[col] = {
                'encoder': le,
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }
            log_progress(f"Encoded {col} with {len(le.classes_)} unique values")
    
    # Create time-based aggregated features using numerical windows instead of time-based
    log_progress("Creating time-based features...")
    temporal_features = ['actual_worth', 'price_per_sqm']
    
    # Group by property type and calculate rolling means
    for col in temporal_features:
        # Calculate 3-month (90-day) rolling mean using approximately 90 records
        data[f'{col}_rolling_mean_3m'] = data.groupby('property_type_en')[col].transform(
            lambda x: x.rolling(window=90, min_periods=1).mean())
        
        # Calculate 6-month (180-day) rolling mean using approximately 180 records
        data[f'{col}_rolling_mean_6m'] = data.groupby('property_type_en')[col].transform(
            lambda x: x.rolling(window=180, min_periods=1).mean())
        
        # Calculate year-over-year change using shift
        data[f'{col}_yoy_change'] = data.groupby('property_type_en')[col].transform(
            lambda x: x.pct_change(periods=365))
    
    # Create location-based features
    log_progress("Creating location-based features...")
    data['avg_area_price'] = data.groupby('area_name_en')['price_per_sqm'].transform('mean')
    data['price_to_area_avg_ratio'] = data['price_per_sqm'] / data['avg_area_price']
    
    # Drop unnecessary columns
    columns_to_drop = [
        'building_name_en', 'project_number', 'project_name_en',
        'master_project_en', 'rooms_en', 'has_parking'
    ]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Save to CSV
    log_progress(f"Saving preprocessed dataset to {output_file}...")
    data.to_csv(output_file, index=False)
    log_progress(f"Dataset saved successfully")
    
    end_time = time()
    processing_time = round(end_time - start_time, 2)
    log_progress(f"Preprocessing completed in {processing_time} seconds")
    
    return data, encoders

def get_training_features(preprocessed_data):
    """
    Extract features suitable for machine learning models.
    """
    feature_columns = [
        'year', 'month', 'quarter', 'day_of_week',
        'procedure_area', 'price_per_sqm',
        'is_residential', 'is_commercial',
        'property_type_en_encoded', 'reg_type_en_encoded',
        'area_name_en_encoded', 'trans_group_en_encoded',
        'price_per_sqm_rolling_mean_3m', 'price_per_sqm_rolling_mean_6m',
        'price_per_sqm_yoy_change',
        'avg_area_price', 'price_to_area_avg_ratio'
    ]
    
    return preprocessed_data[feature_columns]

def print_dataset_info(data):
    """
    Print useful information about the preprocessed dataset.
    """
    log_progress("Generating dataset summary...")
    print("\nDataset Information:")
    print(f"Time range: {data['instance_date'].min()} to {data['instance_date'].max()}")
    print(f"Number of transactions: {len(data)}")
    print("\nProperty Types:")
    print(data['property_type_en'].value_counts())
    print("\nTransaction Types:")
    print(data['trans_group_en'].value_counts())
    print("\nFeature Statistics:")
    print(data[['price_per_sqm', 'procedure_area', 'actual_worth']].describe())

# Example usage
if __name__ == "__main__":
    log_progress("Starting script execution...")
    
    # Read the CSV file
    input_file = 'Transactions.csv'  # Replace with your input file path
    output_file = 'preprocessed_real_estate.csv'
    
    log_progress(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Preprocess the data
    preprocessed_data, encoders = preprocess_dubai_real_estate(df, output_file)
    
    # Get features for modeling
    features = get_training_features(preprocessed_data)
    
    # Print dataset information
    print_dataset_info(preprocessed_data)
    
    log_progress("Script execution completed successfully")