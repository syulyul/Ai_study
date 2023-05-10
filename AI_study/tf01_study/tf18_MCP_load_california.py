import numpy as np
from keras.models import Sequential, load_model
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

# model.save('./_save/tf15_california.h5')   # h5 로 모델 저장

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# ## EarlyStopping
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
#                               verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

# Model Check Point
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath='./_mcp/tf18_california.hdf5'
# )

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=200,      #batch_size 커야 훈련시간 단축됨
#           validation_split=0.2,  # validation data => 0.2 (train 0.6 / test 0.2)
#           callbacks=[earlyStopping, mcp],
#           verbose=1)    
# end_time = time.time() - start_time

# model.save_weights('./_save/tf17_weight_california.h5')

model = load_model('./_mcp/tf18_california.hdf5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss : ', loss)
print('r2스코어 : ', r2)
# print('걸린 시간 : ', end_time)


#===============================================#
# Epoch 406: early stopping
# loss :  0.6052903532981873
# r2스코어 :  0.5507706539539046
# 걸린 시간 :  43.20632314682007

#===============================================#
# loss :  0.5811012983322144
# r2스코어 :  0.5687231625928248