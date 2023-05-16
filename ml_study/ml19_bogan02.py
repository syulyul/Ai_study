import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
# print(data)
# print(data.shape)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
# print(data.shape)

# from sklearn.impute import IterativeImputer   # 실험 중
# from sklearn.experimental import enable_iterative_imputer # object 사용 불가
from sklearn.impute import SimpleImputer

# imputer = SimpleImputer()   # 평균 값으로 대체(default)
# imputer = SimpleImputer(strategy='mean')    # 평균값
# imputer = SimpleImputer(strategy='median')  # 중간값
# imputer = SimpleImputer(strategy='most_frequent')   # 가장 많이 사용된 값
imputer = SimpleImputer(strategy='constant')    # 상수(default = 0)
# imputer = SimpleImputer(strategy='constant', fill_value=777)    # 특정한 값 --> 상수(default = 0) 입력
imputer.fit(data)
data2 = imputer.transform(data)
print(data2)