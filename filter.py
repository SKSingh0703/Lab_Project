import pandas as pd

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

print("Full shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nFull class balance:\n", df["Diabetes_binary"].value_counts(normalize=True))

print("\nAge codes present:\n", df["Age"].value_counts().sort_index())

young = df[df["Age"].isin([1, 2, 3])].copy()   # 18-34
print("\nYoung adult shape:", young.shape)
print("Young adult class counts:\n", young["Diabetes_binary"].value_counts())
print("Young adult class %:\n", young["Diabetes_binary"].value_counts(normalize=True))

young.to_csv("diabetes_young_adults.csv", index=False)