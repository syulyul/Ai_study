import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) #weight의 난수값 조절

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7,  random_state=100, shuffle= True 
)

# kfold
n_splits = 11    # 보통 홀수로 들어감
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)

# Scaler 적용
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x_train = scaler.transform(x_train) # train 은 fit, transform 모두 해줘야 함
x_test = scaler.transform(x_test)   # test 는 transform 만 하면 됨

# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor(random_state=123, n_estimators=1000,
                    learning_rate = 0.1, max_depth = 6, gamma= 1)

# 3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=20,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse')  
          # eval_metric 회귀모델 : rmse, mae, rmsle...
          #             이진분류 : error, auc, logloss...
          #             다중분류 : merror, mlogloss...
          # eval_set : 검증
        

# 4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold)   # cv : corss validation
# print('cv acc : ', score)   # kfold 에 있는 n_splits 숫자만큼 나옴 
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)

r2 = r2_score(y_test, y_predict)
print('cv pred r2 : ', r2)
# cv pred r2 :  0.8184110207883767
