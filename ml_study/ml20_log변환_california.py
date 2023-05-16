import numpy as np
import pandas as pd     # 추가
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


df = pd.DataFrame(x, columns=[datasets.feature_names])
print(df)
print(df.head(10))  # 바꿔야 할 데이터 확인 (수치가 너무 크면 의미가 큰 데이터로 인식하므로)
print(df['Population'])

df['Population'] = np.log1p(df['Population'])
print(df['Population'].head())



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
model = XGBRegressor()

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

r2 = r2_score(y_test, y_predict)
print('cv pred r2 : ', r2)
# cv pred r2 :  0.8161893201063641

'''
##### feature impoartances #######
print(model, " : ", model.feature_importances_)
# 시각화
import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()
'''
# 시각화
import matplotlib.pyplot as plt
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('california Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)

plt.show()

# SVR() 결과 r2 :  -0.01663695941103427
# 결과 r2 :  0.06830124384888547
# my r2스코어 :  0.5346585367965508

# RobustScaler 적용 후 결과 r2 :  0.6873119065345796
# ====================================================
# tree 결과 r2 :  0.612701922946608
# ====================================================
# ensemble 결과 r2 :  0.8114840410530733
# ====================================================
# kfold 결과 r2 :  0.8128621988883818
