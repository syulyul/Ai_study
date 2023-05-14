import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import time

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000 # 임베딩 레이어의 input_dim
)
print(x_train)
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(np.unique(y_train, return_counts=True))   # 데이터 확인 : 답변이 몇 개인지 확인 가능
print(len(np.unique(y_train)))  # 2

# 최대 길이와 평균 길이
print('리뷰의 최대 길이 : ', max(len(i) for i in x_train))
print('리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))


# pad_sequences
x_train = pad_sequences(x_train, padding='pre',
                       maxlen=100)
x_test = pad_sequences(x_test, padding='pre',
                      maxlen=100)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim = 100))
model.add(LSTM(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# [실습] 코드 완성하기

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics='accuracy')

## EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Model Check Point
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    # save_weights_only=False,
    save_best_only=True,
    filepath='./_mcp/embedding_imdb.hdf5'
)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',  # mode='auto'가 기본, patience=100 --> 100부터 줄여나가기
                              verbose=1, restore_best_weights=True) # restore_best_weights --> default 는 False 이므로 True 로 꼭!!! 변경!!!  

start_time = time.time()    # 시작 시간

model.fit(x_train, y_train, epochs=100, 
          batch_size=32, validation_split=0.2,
          callbacks=[earlyStopping, mcp])  # early Stopping 을 할 것

end_time = time.time() - start_time


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# Epoch 13: early stopping
# loss :  0.4743201434612274
# acc :  0.777999997138977
