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
    shuffle=True        # 데이터를 섞을 것인지 정함 (shuffle : True 가 default)
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim = 1))
model.add(Dense(50))
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
# loss :  1.276703187613748e-10
# r2스코어 :  0.99999999999226