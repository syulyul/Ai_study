import numpy as np
from keras.models import Sequential, load_model # load_model 추가
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from sklearn.datasets import load_boston  # 윤리적 문제로 제공 안 됨
from sklearn.datasets import fetch_california_housing
import time


# 1. 데이터
# datasets = load_boston()
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)   # DESCR 세부적으로 보겠다
# 속성 정보:
# - 그룹의 중위수 소득
# - 그룹의 주택 연령 중위수
# - 가구당 평균 객실 수
# - 평균 가구당 침실 수
# - 모집단 그룹 모집단
# - 평균 가구원수
# - Latitude 그룹 위도
# - 경도 그룹 경도

print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

print(x_train.shape)    # (14447, 8)    # 8(열) - input_dim
print(y_train.shape)    # (14447,)


# 2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim = 8))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))     # 주택 가격

# model.save('./_save/tf15_california.h5')   # h5 로 모델 저장(대용량 데이터를 저장하기 위한 파일 형식 중 하나)
# model = load_model('./_save/tf15_california.h5')

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# ## EarlyStopping
# from keras.callbacks import EarlyStopping

# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
#                               verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=200,      #batch_size 커야 훈련시간 단축됨
#           validation_split=0.2,  # validation data => 0.2 (train 0.6 / test 0.2)
#           callbacks=[earlyStopping],
#           verbose=1)    
# end_time = time.time() - start_time

model = load_model('./_save/tf16_california.h5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss : ', loss)
print('r2스코어 : ', r2)
# print('걸린 시간 : ', end_time)


#===============================================#
# epochs : 500
# loss :  0.6172417998313904
# r2스코어 :  0.5419006881010481
# 걸린 시간 :  49.09269857406616

# patient : 100 / epochs : 5000
# Epoch 378: early stopping
# loss :  0.6085953116416931
# r2스코어 :  0.5483176328192945
# 걸린 시간 :  37.19138145446777

# patient : 50 / epochs : 5000
# Epoch 231: early stopping
# loss :  0.6172729134559631
# r2스코어 :  0.5418776626257056
# 걸린 시간 :  23.557950019836426

#===============================================#
# val_loss: 0.6858
# Epoch 203: early stopping
# loss :  0.6270699501037598
# r2스코어 :  0.534606563820972
# 걸린 시간 :  20.832276105880737

# load_model 로 훈련한 결과
# Epoch 1043: early stopping
# loss :  0.5896702408790588
# r2스코어 :  0.5623635897536061
# 걸린 시간 :  102.61732363700867