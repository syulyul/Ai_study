import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR    # C = cassifier(분류), R = Regresser(회귀)
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델구성
model = Sequential()    # Sequential() --> compile, 훈련 다 해야됨
model.add(Dense(1, input_dim=2, 
                activation='sigmoid'))  # sklearn 의 perceptron()과 동일

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


# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 : [[0.5091916 ]
#  [0.6881192 ]
#  [0.23322645]
#  [0.3927872 ]]
# 모델의 loss :  0.7600290179252625
# acc :  0.5
