# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_iris()  # 다중분류
x = datasets['data']
y = datasets.target

# kfold
n_splits = 5    # 보통 홀수로 들어감
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler 적용 (?)
scaler = MinMaxScaler()
scaler.fit(x)                 # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x) # train 은 fit, transform 모두 해줘야 함


# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x, y)


# 4. 평가, 예측
result = model.score(x, y)
print('결과 acc : ', result)


# SVC() 결과 acc :  0.9777777777777777
# LinearSVC() 결과 acc :  0.9777777777777777
# my 결과 acc :  1.0

# MinMaxScaler() 결과 acc :  0.9777777777777777
# ===============================================
# tree 결과 acc :  0.9555555555555556
# ===============================================
# ensemble 결과 acc :  0.9555555555555556
# ===============================================
# KFold 결과 acc :  1.0