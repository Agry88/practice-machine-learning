import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getCleanDataset import getCleanDataset

# 判斷是否為Engineer職位

# 讀取資料
# 載入資料
df = pd.read_excel("./processed_data.xlsx", sheet_name='Sheet1', usecols="A:I")

# qcut salary_in_usd
df['salary_in_usd_level'] = pd.qcut(df['salary_in_usd'], q=2, labels=False)

# 定義 X 和 Y
X = df.drop(labels=['salary_in_usd_level'] ,axis=1)
y = df['salary_in_usd_level'].values

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
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
classifier.fit(X_train, y_train)

# 模型預測
y_pred = classifier.predict(X_test)
# 預測成功的比例
print('訓練集: ', classifier.score(X_train,y_train))
print('測試集: ', classifier.score(X_test,y_test))

## 匯出圖表 
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_test,y_test, clf=classifier,legend=2)
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 混淆局鎮
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(X_test))
print(cm)

# Precision, Recall, F1-scroe  
from sklearn.metrics import classification_report
print(classification_report(y_test, classifier.predict(X_test)))

# AUC, ROC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict(X_test))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()