import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getCleanDataset import getCleanDataset

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
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy')

# SMOTE過採樣
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# 定義參數網格，進行網格搜索
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 1000],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('best_accuracy: ', best_accuracy) # best_accuracy:  0.8252508361204013
print('best_parameters: ', best_parameters) # best_parameters:  {'criterion': 'entropy', 'n_estimators': 300}

# 使用交叉驗證模型，並計算準確度與標準差
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=grid_search, X=X_train, y=y_train, cv=10)
print('accuracies: ', accuracies) # accuracies:  [0.89130435 0.7173913  0.91304348 0.89130435 0.82608696 0.84782609 0.76086957 0.80434783 0.8 0.86666667]
print('mean: ', accuracies.mean()) # mean:  0.8318840579710145
print('std: ', accuracies.std()) # 0.9078947368421053

# 預測成功的比例
print('訓練集: ', grid_search.score(X_train,y_train))
print('測試集: ', grid_search.score(X_test,y_test))

# 混淆局鎮
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, grid_search.predict(X_test))
print(cm)

# Precision, Recall, F1-scroe  
from sklearn.metrics import classification_report
print(classification_report(y_test, grid_search.predict(X_test)))

## 匯出圖表 
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_test,y_test, clf=grid_search,legend=2)
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# AUC, ROC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict(X_test))
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

# AUC number
print(roc_auc)
