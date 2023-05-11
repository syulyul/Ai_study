import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout  #
from keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 모델링 하기 전에 reshape 해주기 (차원 늘리기 : 합성곱 레이어 전 차원 수 맞춰주기)
# reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# [실습] 빠르게

# 정규화 (Normalization)    --> 0 ~ 1 사이로 숫자 변환 (정규화 해야 잘 나옴)
x_train, x_test = x_train/255.0, x_test/255.0


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(2, 2))   # 차원을 반으로 줄여줌
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))   #
# model.add(Dropout(0.2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=256)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

#=====================결과======================#
#MaxPooling2D(2,2) 레이어 1, Dropout(0.2)
# loss :  0.25890305638313293
# acc :  0.916700005531311

# MaxPooling2D(2, 2)
# Dropout(0.3)
# loss :  0.22996017336845398
# acc :  0.9229999780654907