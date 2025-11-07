import pandas as pd
import glob

path = r"C:\Users\sergi\Downloads\MachineLearningCSV\MachineLearningCVE"
files = glob.glob(path + r"\*.csv")

print("Find files:", len(files))
print(files)

if len(files) == 0:
    raise Exception("Not find CSV files")

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.to_csv("cicids2017.csv", index=False)

print("DONE → cicids2017.csv created!")
