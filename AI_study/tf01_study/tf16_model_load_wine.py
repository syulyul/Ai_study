# [실습] loss = 'sparse_categorical_crossentropy' 를 사용하여 분석
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import time

# 1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
#  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
#  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']


x = datasets['data']
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(y_test)


# 2. 모델 구성
# model = Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(50))
# model.add(Dense(40))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))   # 종류 : 0, 1, 2 이므로 3가지

# model.save('./_save/tf15_wine.h5')   # h5 로 모델 저장
# model = load_model('./_save/tf15_wine.h5')

# 3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])

# from keras.callbacks import EarlyStopping

# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
#                               verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=100,
#                  verbose=1, validation_split=0.2,
#                  callbacks=[earlyStopping])
# end_time = time.time() - start_time
# print('걸린 시간 : ', end_time)

model = load_model('./_save/tf16_wine.h5')

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


# 걸린 시간 :  2.5759541988372803
# loss :  0.09535881876945496
# acc :  0.9629629850387573




#===============================================#
# patience : 50, epochs : 500


# patience : 100, epochs : 500


# patience : 500, epochs : 5000
