# [실습]
# 1. R2score 를 음수가 아닌 0.5 이하로 만들어 보기
# 2. 데이터는 건드리지 말기
# 3. 레이어는 인풋, 아웃풋 포함 7개(히든레이어 5개 이상) 이상으로 만들기
# 4. batch_size=1
# 5. 히든레이어의 노드 개수는 10개 이상 100개 이하
# 6. train_size = 0.7
# 7. ephocs = 100 이상
# 8. [실습시작]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x_train, x_test, y_train, y_test = train_test_split(    # 사이킷런에 있는 train_test_split
    x, y,               # x, y 데이터
    test_size=0.3,      # test의 사이즈 보통 30%
    train_size=0.7,     # train의 사이즈는 보통 70%
    random_state=100,   # 데이터를 난수값에 의해 추출한다는 의미, 중요한 하이퍼 파라미터 (random_state : 난수 값 지정)
    shuffle=False        # 데이터를 섞을 것인지 정함 (shuffle : True 가 default)
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim = 1))      # 노드 수, 히든 레이어 수 영향
model.add(Dense(100))       # 히든 레이어 많이 쌓으면 안됨
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss ='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size = 1)


# 4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)              # 훈련 잘 되어있는지 체크

y_predict = model.predict(x)

##### R2score #####
from sklearn.metrics import r2_score, accuracy_score    # metrics --> 분석(r2 회귀, accuracy 분류)
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

# result(1)
# loss : 62.93217849731445
# r2스코어 : 0.1696187048801573
# 노드수를 늘리고 히든레이어의 개수를 늘리면 r2score 가 좋지 않음