# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1))   # 입력층
model.add(Dense(5))                # 히든 레이어 1
model.add(Dense(7))                # 히든 레이어 2
model.add(Dense(5))                # 히든 레이어 3
model.add(Dense(1))                 # 출력층

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)         # epochs(훈련량) 높이면 시간 늘어남

# 4. 예측, 평가
loss = model.evaluate(x, y)         # 평가
print('loss : ', loss)              # loss :  1.8947806851624636e-14

result = model.predict([4])         # 예측 (4 를 예측해보자. 4는 x result 는 y)
print('4의 예측값 : ', result)      # 4의 예측값 :  [[4.000001]]