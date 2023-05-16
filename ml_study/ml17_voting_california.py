import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# [실습] 성능 향상시키기
# 2. 모델
xgb = XGBRegressor()
                    # ml15_gridSearchCV_xgb_iris 에서 찾은 최적의 파라미터 넣기
lgbm = LGBMRegressor()
cat = CatBoostRegressor(depth = 9, l2_leaf_reg = 5, learning_rate = 0.1)

model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    # voting='soft', 필요없음
    n_jobs=-1
)

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)
# print('voting 결과 : ', score)

regressor = [cat, xgb, lgbm]
for model in regressor :
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# default
# CatBoostRegressor 정확도 :  0.8492
# XGBRegressor 정확도 :  0.8287 (default 가 좋음)
# LGBMRegressor 정확도 :  0.8365

# catboost
# depth = 9, l2_leaf_reg = 5, learning_rate = 0.1
# CatBoostRegressor 정확도 :  0.8543 (최적)