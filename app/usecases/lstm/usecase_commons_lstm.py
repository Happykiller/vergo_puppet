#app\usecases\lstm\usecase_commons_lstm.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from app.services.logger import logger

def preprocess_input_data(df: pd.DataFrame, scaler, coco_encoder) -> pd.DataFrame:
    # Handle missing values by forward and backward filling
    df = df.infer_objects()
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Extract temporal features
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Select numerical features
    numeric_features = scaler.feature_names_in_
    df_numeric = df[numeric_features].copy()

    # Convert columns explicitly to numeric
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Scale numeric data using the provided scaler
    df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)

    # One-hot encode 'coco' if present
    if 'coco' in df.columns:
        coco_encoded = coco_encoder.transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=coco_encoder.get_feature_names_out())
        # Combine scaled numeric and encoded categorical data
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled

    # Check for and replace any remaining NaN values
    if df_processed.isnull().values.any():
        logger.warning("NaN values found in df_processed after preprocessing. Replacing them with 0.")
        df_processed = df_processed.fillna(0)

    return df_processed

def preprocess_data(df: pd.DataFrame):
    # Sort data by 'time' and handle missing values
    df = df.sort_values('time')
    df = df.set_index('time')

    # Convert object columns to appropriate types
    df = df.infer_objects(copy=False)

    # Select numeric columns for interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Interpolate missing values in numeric columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # Fill remaining missing values in all columns
    df = df.ffill().bfill()

    # Explicitly infer types after filling operations
    df = df.infer_objects(copy=False)

    # Reset index
    df = df.reset_index()

    # Continue with further preprocessing steps
    # Extract temporal features
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Store the target variable 'temp'
    y_temp = df['temp'].copy()

    # Remove 'temp' from the input data
    df = df.drop(columns=['temp'])

    # Select numeric features (excluding 'temp')
    numeric_features = ['dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres', 'hour', 'dayofweek', 'month']
    numeric_features = [col for col in numeric_features if col in df.columns]
    df_numeric = df[numeric_features].copy()

    # Convert columns explicitly to numeric
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Normalize numeric data
    scaler = MinMaxScaler()
    df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Normalize the target variable 'temp'
    y_temp = np.array(y_temp).reshape(-1, 1)
    target_scaler = MinMaxScaler()
    y_temp_scaled = target_scaler.fit_transform(y_temp)

    # One-hot encode 'coco' if present
    if 'coco' in df.columns:
        coco_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        coco_encoded = coco_encoder.fit_transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=[f'coco_{int(i)}' for i in coco_encoder.categories_[0]])
        # Combine scaled numeric and encoded categorical data
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled

    # Check for and replace any remaining NaN values
    if df_processed.isnull().values.any():
        logger.warning("NaN values found in df_processed after preprocessing. Replacing them with 0.")
        df_processed = df_processed.fillna(0)
    
    return df_processed, y_temp_scaled, scaler, target_scaler, coco_encoder
    
def prepare_sequences(df_processed: pd.DataFrame, y_temp_scaled, sequence_length: int = 24):
    data = df_processed.values
    target = y_temp_scaled
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
    return X, y

def preprocess_input_data(df: pd.DataFrame, scaler, coco_encoder) -> pd.DataFrame:
    # Sort data by 'time' and handle missing values
    df = df.sort_values('time')
    df = df.set_index('time')

    # Convert object columns to appropriate types
    df = df.infer_objects(copy=False)

    # Select numeric columns for interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Interpolate missing values in numeric columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # Fill remaining missing values in all columns
    df = df.ffill().bfill()

    # Explicitly infer types after filling operations
    df = df.infer_objects(copy=False)

    # Reset index
    df = df.reset_index()

    # Extract temporal features
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Select numeric features
    numeric_features = scaler.feature_names_in_
    df_numeric = df[numeric_features].copy()
    
    # Convert columns explicitly to numeric
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Scale numeric data using the provided scaler
    df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)
    
    # One-hot encode 'coco' if present
    if 'coco' in df.columns:
        coco_encoded = coco_encoder.transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=coco_encoder.get_feature_names_out())
        # Combine scaled numeric and encoded categorical data
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled
    
    # Check for and replace any remaining NaN values
    if df_processed.isnull().values.any():
        logger.warning("NaN values found in df_processed after preprocessing. Replacing them with 0.")
        df_processed = df_processed.fillna(0)
    
    return df_processed

def inverse_transform_predictions(predictions_normalized, target_scaler):
    predictions_normalized = np.array(predictions_normalized).reshape(-1, 1)
    predictions_inverse = target_scaler.inverse_transform(predictions_normalized).flatten()
    return predictions_inverse
