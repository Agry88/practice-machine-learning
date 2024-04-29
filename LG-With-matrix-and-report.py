import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getCleanDataset import getCleanDataset

# 判斷是否為Engineer職位

# 讀取資料
data_df = getCleanDataset()

# 只留下 job_title, salary_in_usd, 以及job_title的dummy variables
data_df = data_df.drop(labels=['work_year', 'experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size'], axis=1)
dummy = pd.get_dummies(data_df['job_title'])
data_df = pd.concat([data_df, dummy], axis=1)
data_df = data_df.drop(labels=['job_title'], axis=1)

# 加一個欄位表示是否為Engineer職位
# 檢查該列的Key是否包含engineer並且值為1
data_df = data_df.assign(is_engineer = lambda x: (x.filter(like='Engineer').eq(1).any(1)).astype(int))

print(data_df)

# 定義 X 和 Y
X = data_df.drop(labels=['is_engineer'] ,axis=1)
y = data_df['is_engineer'].values

# 進行資料集分類
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

# 特徵縮放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# 降維
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 模型擬合
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression()

# 模型預測
logi.fit(X_train, y_train)
# 預測成功的比例
print("Logistic Regression:")
print('訓練集: ', logi.score(X_train,y_train))
print('測試集: ', logi.score(X_test,y_test))

# 混淆局鎮
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, logi.predict(X_test))
print(cm)

# Precision, Recall, F1-Score
from sklearn.metrics import classification_report
print(classification_report(y_test, logi.predict(X_test)))

