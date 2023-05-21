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

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# [실습] 4시까지 성능향상시키기!!! gridSearchCV를 통한 하이퍼파라미터 적용!!!

#2. 모델
xgb = XGBRegressor()
lgbm = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    n_jobs=-1
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)
# print('보팅 결과 : ', score)

regressors = [cat, xgb, lgbm]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# joblib 저장
import joblib
path = './_data/' # 패스 경로 잡기
joblib.dump(model, path+'project01.dat') # 저장이름 설정

# CatBoostRegressor 정확도 :  0.8492
# XGBRegressor 정확도 :  0.8287
# LGBMRegressor 정확도 :  0.8365