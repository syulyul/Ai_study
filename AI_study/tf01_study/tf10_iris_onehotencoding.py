import numpy as np
from keras.models import Sequential
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

# x = datasets.data --> x = datasets['data'] 와 같음
x = datasets['data']    
y = datasets.target     
print(x.shape, y.shape) # (150, 4) (150,)  --> input_dim : 4 / 다중분류 : Class(분류)

### one hot encoding ###
from keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape)     # (105, 4) (105, 3)
print(x_test.shape, y_test.shape)       # (45, 4) (45, 3)
print(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=4))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))   # 다중 분류

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # sparse_categorical_crossentropy : one_hot_encoding 하지 않고 가능
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100) # 훈련 시킴
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)    # acc 1.0 -> 과적합일 확률 높음

# 걸린 시간 :  2.4981889724731445
# loss :  0.041628070175647736
# acc :  0.9777777791023254

### argmax 로 accuracy score 구하기
y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)
y_test = y_test.argmax(axis=1)
argmax_acc = accuracy_score(y_test, y_predict)
print('argmax_acc', argmax_acc)