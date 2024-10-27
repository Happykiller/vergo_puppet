import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from app.services.logger import logger

def preprocess_input_data(df: pd.DataFrame, scaler, coco_encoder) -> pd.DataFrame:
    # Gestion des valeurs manquantes
    df = df.infer_objects()
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Extraction des caractéristiques temporelles
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Sélection des caractéristiques numériques
    numeric_features = scaler.feature_names_in_
    df_numeric = df[numeric_features].copy()

    # Conversion explicite en numérique
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Normalisation des données numériques
    df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)

    # Encodage one-hot de 'coco' s'il existe
    if 'coco' in df.columns:
        coco_encoded = coco_encoder.transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=coco_encoder.get_feature_names_out())
        # Combinaison des données
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled

    # Vérifier et traiter les NaN restants
    if df_processed.isnull().values.any():
        logger.warning("Des valeurs NaN sont présentes dans df_processed après le prétraitement. Elles seront remplacées par 0.")
        df_processed = df_processed.fillna(0)

    return df_processed

def preprocess_data(df: pd.DataFrame):
    # Gestion des valeurs manquantes
    df = df.sort_values('time')
    df = df.set_index('time')

    # Convertir les objets en types appropriés
    df = df.infer_objects(copy=False)

    # Sélectionner les colonnes numériques pour l'interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Interpolation sur les colonnes numériques
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # Remplissage des valeurs manquantes pour toutes les colonnes
    df = df.ffill().bfill()

    # Convertir explicitement les types après les opérations de remplissage
    df = df.infer_objects(copy=False)

    # Réinitialiser l'index
    df = df.reset_index()

    # Suite du prétraitement...
    # Extraction des caractéristiques temporelles
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Stocker la variable cible 'temp'
    y_temp = df['temp'].copy()

    # Supprimer 'temp' des données d'entrée
    df = df.drop(columns=['temp'])

    # Sélection des caractéristiques numériques (excluant 'temp')
    numeric_features = ['dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres', 'hour', 'dayofweek', 'month']
    numeric_features = [col for col in numeric_features if col in df.columns]
    df_numeric = df[numeric_features].copy()

    # Conversion explicite en numérique
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Normalisation des données numériques
    scaler = MinMaxScaler()
    df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Normalisation de la variable cible 'temp'
    y_temp = np.array(y_temp).reshape(-1, 1)
    target_scaler = MinMaxScaler()
    y_temp_scaled = target_scaler.fit_transform(y_temp)

    # Encodage one-hot de 'coco' s'il existe
    if 'coco' in df.columns:
        coco_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        coco_encoded = coco_encoder.fit_transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=[f'coco_{int(i)}' for i in coco_encoder.categories_[0]])
        # Combinaison des données
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled

    # Vérifier et traiter les NaN restants
    if df_processed.isnull().values.any():
        logger.warning("Des valeurs NaN sont présentes dans df_processed après le prétraitement. Elles seront remplacées par 0.")
        df_processed = df_processed.fillna(0)
    
    return df_processed, y_temp_scaled, scaler, target_scaler, coco_encoder
    
def prepare_sequences(df_processed: pd.DataFrame,  y_temp_scaled, sequence_length: int = 24):
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
    # Gestion des valeurs manquantes
    df = df.sort_values('time')
    df = df.set_index('time')

    # Convertir les objets en types appropriés
    df = df.infer_objects(copy=False)

    # Sélectionner les colonnes numériques pour l'interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Interpolation sur les colonnes numériques
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # Remplissage des valeurs manquantes pour toutes les colonnes
    df = df.ffill().bfill()

    # Convertir explicitement les types après les opérations de remplissage
    df = df.infer_objects(copy=False)

    # Réinitialiser l'index
    df = df.reset_index()

    # Extraction des caractéristiques temporelles
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Sélection des caractéristiques numériques
    numeric_features = scaler.feature_names_in_
    df_numeric = df[numeric_features].copy()
    
    # Conversion explicite en numérique
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Normalisation des données numériques
    df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)
    
    # Encodage one-hot de 'coco' s'il existe
    if 'coco' in df.columns:
        coco_encoded = coco_encoder.transform(df[['coco']].fillna(-1))
        coco_encoded_df = pd.DataFrame(coco_encoded, columns=coco_encoder.get_feature_names_out())
        # Combinaison des données
        df_processed = pd.concat([df_numeric_scaled.reset_index(drop=True), coco_encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df_numeric_scaled
    
    # Vérifier et traiter les NaN restants
    if df_processed.isnull().values.any():
        logger.warning("Des valeurs NaN sont présentes dans df_processed après le prétraitement. Elles seront remplacées par 0.")
        df_processed = df_processed.fillna(0)
    
    return df_processed

def inverse_transform_predictions(predictions_normalized, target_scaler):
    predictions_normalized = np.array(predictions_normalized).reshape(-1, 1)
    predictions_inverse = target_scaler.inverse_transform(predictions_normalized).flatten()
    return predictions_inverse