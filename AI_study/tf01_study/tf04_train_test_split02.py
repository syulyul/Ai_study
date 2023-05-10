import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

# x = np.array(range(1,21)) #range함수로도 가능

# print(x.shape)
# print(y.shape)
# print(x)

# x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

# x_test = np.array([15,16,17,18,19,20])
# y_test = np.array([15,16,17,18,19,20])

# [실습] x와 y 데이터를 파이선 리스트 스플릿으로 분리하세요.
# x_train = x[0:14]
# y_train = y[0:14]
# x_test = x[14:20]
# y_test = y[14:20]
x_train = x[:14]
y_train = y[:14]
x_test = x[14:]
y_test = y[14:]

print(x_train)
print(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim = 1))
model.add(Dense(50))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss ='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size = 1)


# 4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result  = model.predict([21])
print('21의 예측값 : ', result)