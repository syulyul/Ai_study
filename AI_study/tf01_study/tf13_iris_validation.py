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

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # sparse_categorical_crossentropy : one_hot_encoding 하지 않고 가능
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)    # acc 1.0 -> 과적합일 확률 높음

# 걸린 시간 :  2.4981889724731445
# loss :  0.041628070175647736
# acc :  0.9777777791023254


# 시각화
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import font_manager, rc
# font_path = 'C:/Windows\\Fonts/D2Coding.ttc'
# font = font_manager.FontProperties(fname=font_path).get_name()
font = 'D2Coding'
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red',  # plot : 점
         label='loss')   
plt.plot(hist.history['val_loss'], marker='.', c='blue',
         label='val_loss')
# plt.title('Loss & Val_loss')
plt.title('로스 값과 검증로스 값')
plt.ylabel('로스')
plt.xlabel('훈련량')
plt.legend()  # 빈 공간에 라벨 표시(레이블)
plt.show()