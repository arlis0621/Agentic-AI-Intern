# generate_dataset.py

import numpy as np
import pandas as pd

# 1) Fix random seed for reproducibility
np.random.seed(42)

# 2) Generate 200 samples from Normal(μ=52, σ=9)
normal_values = np.random.normal(loc=52, scale=9, size=200)

# 3) Generate 200 samples from Gamma(shape=2, scale=10)
gamma_values = np.random.default_rng().gamma(shape=2, scale=10, size=200)

# 4) Build DataFrame
df = pd.DataFrame({
    "value": normal_values,
    "age": gamma_values
})

# 5) Round “age” to integer
df["age"] = df["age"].round().astype(int)

# 6) Save to CSV (in the project root)
df.to_csv("sample_data.csv", index=False)

print("Wrote sample_data.csv with columns: ")
print(df.describe())
