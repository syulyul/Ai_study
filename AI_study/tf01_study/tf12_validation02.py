import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

# x = np.array(range(1,21)) #range함수로도 가능

# print(x.shape)
# print(y.shape)
# print(x)

x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

x_val = np.array([13,14,15,16])     # validation 필요한 이유 : 트레인 셋을 검증
y_val = np.array([13,14,15,16])

x_test = np.array([17,18,19,20])
y_test = np.array([17,18,19,20])


# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim = 1))
model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련  
model.compile(loss ='mse' , optimizer='adam')   # 계산하는 방법
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,  # 조정
          validation_data=[x_val, y_val])


# 4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result  = model.predict([21])
print('21의 예측값 : ', result)


## history_val_loss 출력
print('=========================================')
print(hist)
# print(hist.history)
print(hist.history['val_loss'])


## loss 와 val_loss 시각화
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from matplotlib import font_manager, rc
# font_path = 'D2Coding'
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red',  # plot : 점
         label='loss')   
plt.plot(hist.history['val_loss'], marker='.', c='blue',
         label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()  # 빈 공간에 라벨 표시(레이블)
plt.show()