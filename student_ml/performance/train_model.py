import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os   # ← add this line

def train():
    # Use the REAL filename from Kaggle
    csv_filename = 'Student_Performance.csv'
    
    # Optional: make it more robust with full path
    # csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), csv_filename)
    # df = pd.read_csv(csv_path)
    
    df = pd.read_csv(csv_filename)   # ← change this line to exactly this
    
    print(f"Successfully loaded {df.shape[0]} rows from {csv_filename}")
    print("Columns:", df.columns.tolist())
    
    # The rest of your code stays the same...
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(
        df['Extracurricular Activities']
    )
    
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    print("Train score:", model.score(X_train, y_train))
    print("Test score :", model.score(X_test, y_test))
    
    joblib.dump(model, 'performance/model.pkl')
    print("Model saved as performance/model.pkl 🎉")