# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 3. wine
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_wine()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# 2. 모델
# model = SVC()
model = LinearSVC()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

# SVC() 결과 acc :  0.5555555555555556
# LinearSVC() 결과 acc :  0.7222222222222222
# my 결과 acc :  0.9259259104728699