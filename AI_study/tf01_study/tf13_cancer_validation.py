import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score    # 적합도 / 정확도
from sklearn.datasets import load_breast_cancer
import time

# 이진분류
# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)  --> input_dim : 30

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))   # 이진분류는 무조건 아웃풋 레이어의 활성화 함수를 sigmoid 로 해줘야 함

# 3. 컴파일 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', 
#               metrics='mse')
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', 'mse'])

start_time = time.time()    # 시작 시간
hist = model.fit(x_train, y_train, epochs=100, batch_size=200,
          verbose=1, validation_split=0.2)        # verbose : 0 일 경우 표시 안 됨 / 2 일 경우 막대기 없어짐
end_time = time.time() - start_time     # 걸린 시간


# 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
loss, acc, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
print('loss : ', loss)
print('acc : ', acc)
print('mse : ', mse)

# [실습] accuracy_score 를 출력하라
# y_predict 반올림하기

# np.where
'''
y_predict = np.where(y_predict > 0.5, 1, 0)
acc = accuracy_score(y_test, y_predict)
print('loss : ', loss)  # loss :  [0.23969072103500366, 0.06037144735455513] --> loss, mse
print('acc : ', acc)    # acc :  0.9239766081871345
'''
# np.round
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('loss : ', loss)  # loss :  [0.26926353573799133, 0.056343384087085724] --> loss, mse
print('acc : ', acc)    # acc :  0.935672514619883
print('걸린 시간 : ', end_time)
# 걸린 시간 :  1.1122465133666992


# 걸린 시간 :  1.1031553745269775
# loss :  [0.35699227452278137, 0.9415204524993896, 0.06094878911972046]
# acc :  0.9415204678362573


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