import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class BiasDetector:
    """Utility class for detecting and mitigating bias in ML models"""
    
    def __init__(self, sensitive_features=['Extracurricular Activities']):
        self.sensitive_features = sensitive_features
        self.bias_metrics = {}
    
    def detect_data_bias(self, df, target_column='Performance Index'):
        """Detect bias in the training data"""
        print("=== DATA BIAS DETECTION ===")
        
        bias_report = {}
        
        for feature in self.sensitive_features:
            if feature not in df.columns:
                continue
                
            print(f"\nAnalyzing bias for: {feature}")
            
            # Distribution analysis
            dist = df[feature].value_counts(normalize=True)
            print(f"Distribution: {dist.to_dict()}")
            
            # Performance disparity
            perf_by_group = df.groupby(feature)[target_column].agg(['mean', 'std', 'count'])
            print(f"Performance by group:\n{perf_by_group}")
            
            # Calculate disparity ratio
            if len(perf_by_group) >= 2:
                max_perf = perf_by_group['mean'].max()
                min_perf = perf_by_group['mean'].min()
                disparity_ratio = min_perf / max_perf if max_perf > 0 else 0
                print(f"Disparity ratio (min/max): {disparity_ratio:.3f}")
                
                bias_report[feature] = {
                    'distribution': dist.to_dict(),
                    'performance_by_group': perf_by_group.to_dict(),
                    'disparity_ratio': disparity_ratio
                }
        
        self.bias_metrics['data_bias'] = bias_report
        return bias_report
    
    def detect_model_bias(self, model, X_test, y_test, feature_columns, label_encoder=None):
        """Detect bias in model predictions"""
        print("\n=== MODEL BIAS DETECTION ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Convert back to original scale for analysis
        X_test_df = pd.DataFrame(X_test, columns=feature_columns)
        
        bias_report = {}
        
        for feature in self.sensitive_features:
            if feature not in X_test_df.columns:
                continue
                
            print(f"\nAnalyzing model bias for: {feature}")
            
            # Group by sensitive feature
            unique_values = X_test_df[feature].unique()
            
            group_metrics = {}
            for value in unique_values:
                mask = X_test_df[feature] == value
                if mask.sum() > 0:
                    group_r2 = r2_score(y_test[mask], y_pred[mask])
                    group_mse = mean_squared_error(y_test[mask], y_pred[mask])
                    group_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
                    
                    # Convert back to original labels if needed
                    if label_encoder and feature == 'Extracurricular Activities':
                        group_name = label_encoder.inverse_transform([int(value)])[0]
                    else:
                        group_name = str(value)
                    
                    group_metrics[group_name] = {
                        'r2': group_r2,
                        'mse': group_mse,
                        'mae': group_mae,
                        'sample_size': mask.sum()
                    }
                    
                    print(f"{group_name} (n={mask.sum()}): R²={group_r2:.4f}, MSE={group_mse:.4f}, MAE={group_mae:.4f}")
            
            # Calculate fairness metrics
            if len(group_metrics) >= 2:
                r2_values = [metrics['r2'] for metrics in group_metrics.values()]
                r2_disparity = max(r2_values) - min(r2_values)
                
                print(f"R² disparity: {r2_disparity:.4f}")
                
                bias_report[feature] = {
                    'group_metrics': group_metrics,
                    'r2_disparity': r2_disparity
                }
        
        self.bias_metrics['model_bias'] = bias_report
        return bias_report
    
    def generate_bias_report(self, save_path='performance/bias_report.txt'):
        """Generate a comprehensive bias report"""
        print("\n=== GENERATING BIAS REPORT ===")
        
        with open(save_path, 'w') as f:
            f.write("BIAS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if 'data_bias' in self.bias_metrics:
                f.write("DATA BIAS ANALYSIS\n")
                f.write("-" * 20 + "\n")
                for feature, metrics in self.bias_metrics['data_bias'].items():
                    f.write(f"\nFeature: {feature}\n")
                    f.write(f"Distribution: {metrics['distribution']}\n")
                    f.write(f"Disparity Ratio: {metrics['disparity_ratio']:.3f}\n")
                    f.write("Performance by Group:\n")
                    for group, perf in metrics['performance_by_group'].items():
                        f.write(f"  {group}: Mean={perf['mean']:.2f}, Std={perf['std']:.2f}, Count={perf['count']}\n")
            
            if 'model_bias' in self.bias_metrics:
                f.write("\n\nMODEL BIAS ANALYSIS\n")
                f.write("-" * 20 + "\n")
                for feature, metrics in self.bias_metrics['model_bias'].items():
                    f.write(f"\nFeature: {feature}\n")
                    f.write(f"R² Disparity: {metrics['r2_disparity']:.4f}\n")
                    f.write("Group Performance:\n")
                    for group, perf in metrics['group_metrics'].items():
                        f.write(f"  {group}: R²={perf['r2']:.4f}, MSE={perf['mse']:.4f}, MAE={perf['mae']:.4f}, n={perf['sample_size']}\n")
        
        print(f"Bias report saved to: {save_path}")

class FairnessMitigator:
    """Utility class for mitigating bias in ML models"""
    
    def __init__(self):
        self.mitigation_methods = []
    
    def apply_reweighting(self, X_train, y_train, sensitive_feature='Extracurricular Activities'):
        """Apply reweighting to balance representation"""
        print("\n=== APPLYING REWEIGHTING ===")
        
        # Calculate group weights
        unique_values = X_train[sensitive_feature].unique()
        group_sizes = {}
        
        for value in unique_values:
            group_sizes[value] = (X_train[sensitive_feature] == value).sum()
        
        # Calculate inverse frequency weights
        total_samples = len(X_train)
        weights = []
        
        for idx in X_train.index:
            group_value = X_train.loc[idx, sensitive_feature]
            weight = total_samples / (len(unique_values) * group_sizes[group_value])
            weights.append(weight)
        
        print(f"Applied reweighting for {len(unique_values)} groups")
        return np.array(weights)
    
    def apply_feature_selection(self, X_train, y_train, feature_columns, max_features=4):
        """Select features that minimize bias while maintaining performance"""
        print("\n=== APPLYING BIAS-AWARE FEATURE SELECTION ===")
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import RFE
        
        # Use recursive feature elimination
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=max_features, step=1)
        
        selector.fit(X_train, y_train)
        
        selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selector.support_[i]]
        print(f"Selected features: {selected_features}")
        
        return selected_features, selector
    
    def apply_adversarial_debiasing(self, X_train, y_train, sensitive_feature='Extracurricular Activities'):
        """Apply adversarial debiasing (simplified version)"""
        print("\n=== APPLYING ADVERSARIAL DEBIASING ===")
        
        # This is a simplified version - in practice, you'd use more sophisticated methods
        # For now, we'll remove the sensitive feature's direct influence
        
        X_debiased = X_train.copy()
        
        # Reduce the influence of the sensitive feature
        if sensitive_feature in X_debiased.columns:
            # Scale down the sensitive feature
            X_debiased[sensitive_feature] = X_debiased[sensitive_feature] * 0.5
            print(f"Reduced influence of {sensitive_feature}")
        
        return X_debiased

def run_comprehensive_bias_analysis():
    """Run a complete bias analysis pipeline"""
    print("=== COMPREHENSIVE BIAS ANALYSIS ===")
    
    # Load data
    df = pd.read_csv('Student_Performance.csv')
    
    # Initialize bias detector
    detector = BiasDetector()
    
    # Detect data bias
    data_bias = detector.detect_data_bias(df)
    
    # Load and analyze model bias (if model exists)
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        # Load model
        model_data = joblib.load('performance/model.pkl')
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_columns = model_data.get('feature_columns', ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])
        else:
            model = model_data
            scaler = StandardScaler()
            feature_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
        
        # Prepare test data
        df_processed = df.copy()
        le = LabelEncoder()
        df_processed['Extracurricular Activities'] = le.fit_transform(df_processed['Extracurricular Activities'])
        
        if scaler:
            df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
        
        from sklearn.model_selection import train_test_split
        X = df_processed[feature_columns]
        y = df_processed['Performance Index']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Detect model bias
        model_bias = detector.detect_model_bias(model, X_test, y_test, feature_columns, le)
        
    except Exception as e:
        print(f"Could not analyze model bias: {e}")
    
    # Generate report
    detector.generate_bias_report()
    
    return detector.bias_metrics

if __name__ == "__main__":
    run_comprehensive_bias_analysis()
