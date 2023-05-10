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

model.save('./_save/tf15_cancer.h5')   # h5 로 모델 저장

# 3. 컴파일 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', 
#               metrics='mse')
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', 'mse'])

## EarlyStopping
from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
                              verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

start_time = time.time()    # 시작 시간
hist = model.fit(x_train, y_train, epochs=100, batch_size=200,
          verbose=1, validation_split=0.2,
          callbacks=[earlyStopping])        # verbose : 0 일 경우 표시 안 됨 / 2 일 경우 막대기 없어짐
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



#===============================================#
# patience=100, epochs=100
# loss :  0.32834070920944214
# acc :  0.9356725215911865
# mse :  0.06004061922430992
# loss :  0.32834070920944214
# acc :  0.935672514619883
# 걸린 시간 :  3.273170232772827

# patience=50, epochs=100   --> patience : 성능향상이 중지된 에포크 수 동안 성능 향상이 없으면 훈련 중단
# loss :  1.6457053422927856
# acc :  0.8538011908531189
# mse :  0.14049509167671204
# loss :  1.6457053422927856
# acc :  0.8538011695906432
# 걸린 시간 :  3.335972547531128