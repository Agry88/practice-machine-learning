import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getCleanDataset import getCleanDataset
from sklearn.ensemble import RandomForestClassifier

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

# 取得label名稱
feat_labels = data_df.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()