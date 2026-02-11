import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_bias(df):
    """Analyze potential bias in the dataset"""
    print("=== BIAS ANALYSIS ===")
    
    # 1. Check class balance for extracurricular activities
    print("\n1. Extracurricular Activities Distribution:")
    extracurricular_dist = df['Extracurricular Activities'].value_counts()
    print(extracurricular_dist)
    print(f"Balance ratio: {extracurricular_dist.min() / extracurricular_dist.max():.3f}")
    
    # 2. Performance distribution by extracurricular activities
    print("\n2. Performance Index by Extracurricular Activities:")
    perf_by_extra = df.groupby('Extracurricular Activities')['Performance Index'].agg(['mean', 'std', 'count'])
    print(perf_by_extra)
    
    # 3. Check for feature correlation bias
    df_encoded = df.copy()
    df_encoded['Extracurricular Activities'] = LabelEncoder().fit_transform(df_encoded['Extracurricular Activities'])
    
    print("\n3. Feature Correlations with Performance:")
    correlations = df_encoded.corr()['Performance Index'].sort_values(ascending=False)
    print(correlations)
    
    # 4. Check for potential proxy variables
    print("\n4. Checking for Proxy Variables:")
    # High correlation between features might indicate proxy bias
    feature_corr = df_encoded.drop('Performance Index', axis=1).corr()
    high_corr_pairs = []
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            if abs(feature_corr.iloc[i, j]) > 0.7:
                high_corr_pairs.append((feature_corr.columns[i], feature_corr.columns[j], feature_corr.iloc[i, j]))
    
    if high_corr_pairs:
        print("Highly correlated feature pairs (potential proxy bias):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No highly correlated features found")
    
    return correlations, high_corr_pairs

def create_fair_split(df, test_size=0.2, random_state=42):
    """Create a fair train-test split that preserves distribution"""
    # Stratified split based on extracurricular activities to ensure fair representation
    df_encoded = df.copy()
    df_encoded['Extracurricular Activities'] = LabelEncoder().fit_transform(df_encoded['Extracurricular Activities'])
    
    # Create bins for performance index for stratification
    df_encoded['perf_bins'] = pd.cut(df_encoded['Performance Index'], bins=5, labels=False)
    
    # Use multi-level stratification
    X = df_encoded.drop(['Performance Index', 'perf_bins'], axis=1)
    y = df_encoded['Performance Index']
    
    # First split: stratified by extracurricular activities and performance bins
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_encoded[['Extracurricular Activities', 'perf_bins']].apply(lambda x: f"{x[0]}_{x[1]}", axis=1)
    )
    
    return X_train, X_test, y_train, y_test

def train_unbiased_model():
    """Train an unbiased model with fairness considerations"""
    print("=== UNBIASED MODEL TRAINING ===")
    
    # Load data
    csv_filename = 'Student_Performance.csv'
    df = pd.read_csv(csv_filename)
    print(f"Loaded {df.shape[0]} samples from {csv_filename}")
    
    # Analyze bias
    correlations, proxy_pairs = analyze_bias(df)
    
    # Preprocessing
    df_processed = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    df_processed['Extracurricular Activities'] = le.fit_transform(df_processed['Extracurricular Activities'])
    
    # Feature scaling (important for fairness)
    scaler = StandardScaler()
    feature_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 
                      'Sleep Hours', 'Sample Question Papers Practiced']
    
    df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
    
    # Create fair split
    X_train, X_test, y_train, y_test = create_fair_split(df_processed)
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Check distribution preservation
    print("\n=== Distribution Preservation Check ===")
    print("Training set - Extracurricular Activities distribution:")
    train_extra_dist = X_train['Extracurricular Activities'].value_counts(normalize=True)
    print(train_extra_dist)
    
    print("Test set - Extracurricular Activities distribution:")
    test_extra_dist = X_test['Extracurricular Activities'].value_counts(normalize=True)
    print(test_extra_dist)
    
    # Train model with cross-validation for robust evaluation
    print("\n=== MODEL TRAINING ===")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"R² Score: {test_r2:.4f}")
    print(f"MSE: {test_mse:.4f}")
    
    # Fairness evaluation
    print("\n=== FAIRNESS EVALUATION ===")
    # Evaluate performance across different groups
    X_test_original = X_test.copy()
    X_test_original[feature_columns] = scaler.inverse_transform(X_test[feature_columns])
    
    # Group by extracurricular activities (0=No, 1=Yes)
    for group in [0, 1]:
        group_mask = X_test_original['Extracurricular Activities'] == group
        if group_mask.sum() > 0:
            group_r2 = r2_score(y_test[group_mask], y_pred[group_mask])
            group_mse = mean_squared_error(y_test[group_mask], y_pred[group_mask])
            group_name = "No Extracurricular" if group == 0 else "Has Extracurricular"
            print(f"{group_name} (n={group_mask.sum()}): R²={group_r2:.4f}, MSE={group_mse:.4f}")
    
    # Feature importance analysis
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save model and preprocessing objects
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'performance_metrics': {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_mse': test_mse
        }
    }
    
    joblib.dump(model_data, 'performance/model_unbiased.pkl')
    print("\n✅ Unbiased model saved as 'performance/model_unbiased.pkl'")
    
    return model_data

def compare_models():
    """Compare original biased model with unbiased model"""
    print("\n=== MODEL COMPARISON ===")
    
    # Load original model if exists
    try:
        original_model = joblib.load('performance/model.pkl')
        print("Original model loaded")
    except:
        print("Original model not found")
        return
    
    # Train unbiased model
    unbiased_model_data = train_unbiased_model()
    unbiased_model = unbiased_model_data['model']
    
    # Load test data for comparison
    df = pd.read_csv('Student_Performance.csv')
    df_processed = df.copy()
    
    le = LabelEncoder()
    df_processed['Extracurricular Activities'] = le.fit_transform(df_processed['Extracurricular Activities'])
    
    scaler = StandardScaler()
    feature_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 
                      'Sleep Hours', 'Sample Question Papers Practiced']
    df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
    
    X = df_processed[feature_columns]
    y = df_processed['Performance Index']
    
    # Split data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare predictions
    original_pred = original_model.predict(X_test)
    unbiased_pred = unbiased_model.predict(X_test)
    
    original_r2 = r2_score(y_test, original_pred)
    unbiased_r2 = r2_score(y_test, unbiased_pred)
    
    print(f"\nPerformance Comparison:")
    print(f"Original Model R²: {original_r2:.4f}")
    print(f"Unbiased Model R²: {unbiased_r2:.4f}")
    print(f"Performance Difference: {unbiased_r2 - original_r2:.4f}")

if __name__ == "__main__":
    # Train unbiased model
    model_data = train_unbiased_model()
    
    # Optional: Compare with original model
    compare_models()
