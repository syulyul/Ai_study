import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = fetch_california_housing()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5    # 보통 홀수로 들어감
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)    # KFold : 회귀모델 / StratifiedKFold : 분류모델


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
from sklearn.model_selection import RandomizedSearchCV
rf_model = RandomForestRegressor()
model = RandomizedSearchCV(rf_model, param, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)   # refit 는 False 가 default (param 적용해라)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits


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
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_leaf': 3}
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=3)
# best_score :  0.8032891773344165
# model_score :  0.8042504674906866
# 걸린 시간 :  71.0728874206543 초
# cv pred acc :  0.7446577413935147


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
r2 = r2_score(y_test, y_predict)
print('cv pred acc : ', r2)


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