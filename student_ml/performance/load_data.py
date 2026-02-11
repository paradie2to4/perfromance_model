import pandas as pd
import os
from performance.models import StudentPerformance

def run():
    # This finds the CSV no matter where you run the script from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → student_ml/
    csv_path = os.path.join(base_dir, 'Student_Performance.csv')              # or wherever you put it

    print(f"Trying to load CSV from: {csv_path}")   # ← helpful debug line

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        StudentPerformance.objects.create(
            hours_studied     = int(row['Hours Studied']),
            previous_scores   = int(row['Previous Scores']),
            extracurricular   = row['Extracurricular Activities'] == 'Yes',
            sleep_hours       = int(row['Sleep Hours']),
            sample_papers     = int(row['Sample Question Papers Practiced']),
            performance_index = float(row['Performance Index']),
        )

    print(f"Loaded {len(df)} students successfully! 🎉")