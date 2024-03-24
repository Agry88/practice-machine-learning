import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(color_codes=True)

# 載入資料
df = pd.read_excel("./ds_salaries.xlsx", sheet_name='ds_salaries', usecols="A:L")

# 1.移除不必要的欄位, 檢查缺失值, 檢查資料型態
df.drop(columns=['salary', 'salary_currency'], inplace=True)
check_missing = df.isnull().sum() * 100 / df.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)
df.select_dtypes(include='object').nunique()

# 2.敘述統計
print(df.describe())

# 3.特徵縮放
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include='number')), columns=df.select_dtypes(include='number').columns)

# 4.類別特徵轉換
df_dummies = pd.get_dummies(df.select_dtypes(include='object'))

# 5. 訓練與測試資料
from sklearn.model_selection import train_test_split

X = pd.concat([df_scaled, df_dummies], axis=1)
y = df['company_size']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)