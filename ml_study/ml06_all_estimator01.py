import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')   # 경고 무시

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, shuffle=True
)

# Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms : ', allAlgorithms)
print('몇 개? : ', len(allAlgorithms))  # 41

# 3. 출력(평가, 예측)
for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        print(name, '출력 안 됨')