# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 4. california (SVR, LinearSVR)
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = fetch_california_housing()   # 회귀분석
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# 2. 모델
# model = SVR()
model = LinearSVR()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 r2 : ', result)


# SVR() 결과 r2 :  -0.01663695941103427
# 결과 r2 :  0.06830124384888547
# my r2스코어 :  0.5346585367965508