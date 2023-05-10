import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import time

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = datasets.data
x = datasets['data']    
y = datasets.target     
print(x.shape, y.shape) # (150, 4) (150,)  --> input_dim : 4 / 다중분류 : Class(분류)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape)     # (105, 4) (105,)
print(x_test.shape, y_test.shape)       # (45, 4) (45,)
print(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=4))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))   # 다중 분류

# model.save('./_save/tf15_iris.h5')   # h5 로 모델 저장
model = load_model('./_save/tf15_iris.h5')

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # sparse_categorical_crossentropy : one_hot_encoding 하지 않고 가능

from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
                              verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
                 verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)    # acc 1.0 -> 과적합일 확률 높음

# 걸린 시간 :  2.4981889724731445
# loss :  0.041628070175647736
# acc :  0.9777777791023254





#===============================================#
# patience : 50, epochs : 500


# patience : 100, epochs : 500


# patience : 500, epochs : 5000


### load
# 걸린 시간 :  12.78639268875122
# loss :  0.015214063227176666
# acc :  1.0