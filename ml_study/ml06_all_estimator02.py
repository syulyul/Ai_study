import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7, random_state=42, shuffle=True 
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
allalgorithms = all_estimators(type_filter='regressor')
print(len(allalgorithms))   # 55 개

# 3. 출력(평가, 예측)
for (name, algorithm) in allalgorithms:
    try : 
        model = algorithm()
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, "의 정답률 : ", r2)
    except :
        print(name, "출력 안 됨")


# 가장 좋은 모델 
# HistGradientBoostingRegressor 의 정답률 :  0.83795759614234