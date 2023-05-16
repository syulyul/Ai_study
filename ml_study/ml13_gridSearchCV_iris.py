import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_iris()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5    # 보통 홀수로 들어감
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_train)       # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_test) 

# [실습] 파라미터 튜닝 및 정리
param = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]


# 2. 모델
from sklearn.model_selection import GridSearchCV
rf_model = RandomForestClassifier()
model = GridSearchCV(rf_model, param, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)   # refit 는 False 가 default


# 3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


print('최적의 파라미터 : ', model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)       # 가장 좋은 score
print('model_score : ', model.score(x_test, y_test))    # 실제 데이터를 넣었을 때의 socre
print('걸린 시간 : ', end_time, '초')
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_split': 10}
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_split=10)                                                                it=10)
# best_score :  0.95
# model_score :  1.0

# 4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold)   # cv : corss validation
# print('cv acc : ', score)   # kfold 에 있는 n_splits 숫자만큼 나옴 
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
# print('cv pred : ', y_predict)
# cv pred :  [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 1 0 2 2 2 2 2 0 0] # 0.8% 를 제외한 나머지 0.2 %
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)


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
# cv acc :  [0.91666667 1.         0.91666667 0.83333333 1.        ]
# cv pred acc :  0.9333333333333333
# stratifiedKFold cv pred acc :  0.9333333333333333