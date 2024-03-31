import pandas as pd

def getCleanDataset():
  # 載入資料
  df = pd.read_excel("./ds_salaries.xlsx", sheet_name='ds_salaries', usecols="A:L")

  # 移除不必要的欄位, 檢查缺失值, 檢查資料型態
  df.drop(columns=['salary', 'salary_currency'], inplace=True)
  check_missing = df.isnull().sum() * 100 / df.shape[0]
  check_missing[check_missing > 0].sort_values(ascending=False)
  df.select_dtypes(include='object').nunique()

  return df