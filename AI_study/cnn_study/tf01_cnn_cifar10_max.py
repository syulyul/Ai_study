import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar10
import time
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 정규화 (Nomalization) => 0 ~ 1 사이로 숫자 변환
x_train, x_test = x_train/255.0, x_test/255.0

#2. 데이터 모델 구성
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (4,4),
                padding = 'same',
                activation = 'relu',
                input_shape = (32,32,3))) #3은 칼라

model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# EarlyStopping 인스턴스 생성
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# ModelCheckpoint 인스턴스 생성
model_checkpoint = ModelCheckpoint('./save/best_model.h5', monitor='val_loss', mode='min', 
                                   save_best_only=True)

# 2. 모델 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
         callbacks=[early_stopping, model_checkpoint], validation_split=0.2)
end_time = time.time()- start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#print(y_predict)
print('loss : ', loss)
print('acc : ', acc)
print('time : ',end_time)