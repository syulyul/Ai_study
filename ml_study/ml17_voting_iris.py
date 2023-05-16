import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
xgb = XGBClassifier(colsample_bylevel = 0, colsample_bynode = 0, 
                    colsample_bytree = 0, gamma = 4, learning_rate = 0.01, 
                    max_depth = 3, min_child_weight = 0, n_estimators = 100, 
                    reg_alpha = 0, reg_lambda = 1, subsample = 0.2)
                    # ml15_gridSearchCV_xgb_iris 에서 찾은 최적의 파라미터 넣기
lgbm = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='soft',
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

classfiers = [cat, xgb, lgbm]
for model in classfiers :
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# hard, soft 같은 결과
# CatBoostClassifier 정확도 :  1.0000
# XGBClassifier 정확도 :  1.0000
# LGBMClassifier 정확도 :  1.0000