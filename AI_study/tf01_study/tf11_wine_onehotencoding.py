# [실습] one hot encoding 을 사용하여 분석 
#       (세 가지 방법 중 한 가지 사용)
#       (1) keras.utils 의 to_categorical
#       (2) pandas 의 get_dummies
#       (3) sklearn.preprocessing 의 OneHotEncoder

import numpy as np
from keras.models import Sequential
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
print(x.shape, y.shape) # (178, 13) (178,)

### one hot encoding ###
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=13, shuffle=True
)
print(x_train.shape, y_train.shape) # (124, 13) (124, 3)
print(x_test.shape, y_test.shape)   # (54, 13) (54, 3)
print(y_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(100 ,input_dim=13))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# 걸린 시간 :  2.516160011291504
# loss :  0.2077403962612152
# acc :  0.9444444179534912