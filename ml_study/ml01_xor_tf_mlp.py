import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR    # C = cassifier(분류), R = Regresser(회귀)
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# [실습] MLP 모델 구성하여 acc = 1.0 만들기
# 2. 모델구성
model = Sequential()    # compile, 훈련 다 해야됨
model.add(Dense(32, input_dim=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # MLP(multi layer perceptron) 와 동일

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='acc')

model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
loss, acc = result = model.evaluate(x_data, y_data)

y_predict = model.predict(x_data)

print(x_data, '의 예측 결과 :', y_predict)
print('모델의 loss : ', loss)
print('acc : ', acc)

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 : [[0.00640666]
#  [0.99336225]
#  [0.9949134 ]
#  [0.00810985]]
# 모델의 loss :  0.006582422647625208
# acc :  1.0
