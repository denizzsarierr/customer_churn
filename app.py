import pandas as pd


df = pd.read_csv("Telco-Customer-Churn")

df.shape
df.info()
df.head()
df.isna().sum() 