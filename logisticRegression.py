from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getCleanDataset import getCleanDataset
import seaborn as sns

# 使用遠端比例及工作推斷薪資

# 讀取資料
data_df = getCleanDataset()

# 只留下 job_title, salary_in_usd, 以及job_title的dummy variables
data_df = data_df.drop(labels=['work_year', 'experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size'], axis=1)
dummy = pd.get_dummies(data_df['job_title'])
data_df = pd.concat([data_df, dummy], axis=1)
data_df = data_df.drop(labels=['job_title'], axis=1)
print(data_df)

# 定義 X 和 Y
X = data_df.drop(labels=['salary_in_usd'] ,axis=1)
y = data_df['salary_in_usd'].values

# 進行資料集分類
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

# 使用邏輯回歸模型
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression()
logi.fit(X_train, y_train)
print("Logistic Regression:")
print('訓練集: ', logi.score(X_train,y_train))
print('測試集: ', logi.score(X_test,y_test))

# 畫散佈圖
plt.scatter(y_test, logi.predict(X_test))
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()