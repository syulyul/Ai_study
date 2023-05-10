import numpy as np
from keras.models import Sequential     # Keras 패키지에서 Sequential 모델을 가져옴
from keras.layers import Dense          # Keras 패키지에서 Dense 레이어 클래스를 가져옴

# 1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # w값이 (-)일 경우 'mae' 사용(loss, 예측값 모두 좋아짐)
# mse --> (y - y')^2 , |y - y'| --> mae(-값이 나올만 할 경우)
model.fit(x, y, epochs=1000)

# 4. 예측, 평가
loss = model.evaluate(x, y)
print('loss : ', loss)          # loss :  0.40052375197410583

result = model.predict([6])
print('6의 예측값 : ', result)   # 6의 예측값 :  [[5.9980288]]