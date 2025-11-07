import pandas as pd

df = pd.read_csv("cicids2017.csv")

print("\n================ CSV LOADED ================")
print("Shape (rows, cols):", df.shape)

print("\n================ COLUMNS ================")
print(df.columns.tolist())

print("\n================ FIRST 10 ROWS ================")
print(df.head(10))

print("\n================ LABEL DISTRIBUTION ================")
if "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("Warning: Column 'Label' doesn't exist in this CSV!")

print("\n================ NULL VALUES ================")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\n================ DONE ================")
