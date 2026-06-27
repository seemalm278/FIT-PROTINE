import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 500

data = {
    'weight': np.random.randint(40, 150, n_samples),
    'height': np.random.randint(140, 210, n_samples),
    'age': np.random.randint(15, 70, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'workout_type': np.random.choice(['Strength', 'Cardio', 'HIIT'], n_samples),
    'workout_freq': np.random.randint(1, 8, n_samples),
    'fitness_goal': np.random.choice(['Gain', 'Maintain', 'Lose'], n_samples),
    'protein_intake': np.random.randint(30, 200, n_samples),
    'protein_required': np.random.randint(50, 250, n_samples)  # This is what we'll predict
}

df = pd.DataFrame(data)
df.to_csv('protein_fitness_dataset.csv', index=False)
print("Dataset created: protein_fitness_dataset.csv")