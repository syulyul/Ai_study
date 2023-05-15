# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_breast_cancer()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42

)
# kfold
n_splits = 11    # 보통 홀수로 들어감
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_train) # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_test) 



# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold)   # cv : corss validation
# print('cv acc : ', score)   # kfold 에 있는 n_splits 숫자만큼 나옴 
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
# print('cv pred : ', y_predict)
# # cv pred : [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
#  1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 
# 0
#  1 1 1 0 1 1 0 1 0 0 0 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 0 1 
# 0
#  1 1 0] # 0.8% 를 제외한 나머지 0.2 %
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9473684210526315


# 결과
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
# ====================================================================
# cv acc :  [0.97619048 0.95238095 0.95238095 1.         0.95121951 1.
#  0.95121951 0.87804878 0.97560976 0.92682927 0.90243902]       ]