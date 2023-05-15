# [실습] svm 모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_iris()  # 다중분류
x = datasets['data']
y = datasets.target

# print(x.shape, y.shape)     # (150, 4) (150,)
# print('y의 라벨 값 :', np.unique(y))    # y의 라벨 값 : [0 1 2]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# 2. 모델
model = SVC()
# model = LinearSVC()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

# SVC() 결과 acc :  0.9777777777777777
# LinearSVC() 결과 acc :  0.9777777777777777
# my 결과 acc :  1.0