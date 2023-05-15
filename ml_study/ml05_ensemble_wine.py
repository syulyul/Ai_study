# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 3. wine
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_wine()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x_train = scaler.transform(x_train) # train 은 fit, transform 모두 해줘야 함
x_test = scaler.transform(x_test)   # test 는 transform 만 하면 됨


# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

# SVC() 결과 acc :  0.5555555555555556
# LinearSVC() 결과 acc :  0.7222222222222222
# my 결과 acc :  0.9259259104728699

# MinMaxScaler() 결과 acc :  0.9814814814814815

# =================================================
# tree 결과 acc :  0.8518518518518519
# =================================================
# ensemble 결과 acc :  1.0
