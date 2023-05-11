import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar100
import time

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

# 정규화(Normalization)
x_train = x_train/255.0
x_test = x_test/255.0

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same',    # 하이퍼파라미터
                 activation='relu', # 하이퍼파라미터
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax')) # cifar100 이므로 100

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
start_time = time.time()
model.fit(x_train, y_train, epochs=10, 
          batch_size=256)
end_time = time.time() - start_time

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss)
print('acc : ', acc)



# 걸린 시간 :  386.53373312950134
# loss :  2.373387098312378
# acc :  0.39500001072883606