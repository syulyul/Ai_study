import numpy as np
from keras.preprocessing.text import Tokenizer

#1. 데이터
docs = ['재밌어요', '재미없다', '돈 아깝다', '숙면했어요', '최고예요',
        '꼭 봐라', '세 번 봐라', '또 보고싶다', 'n회차 관람', '배우가 잘 생기긴 했어요',
        '발연기에요','추천해요','최악','후회된다','돈 버렸다','글쎄요', '보다 나왔다',
        '망작이다','연기가 어색해요','차라리 기부할걸','다음편 나왔으면 좋겠다',
        '다른 거 볼걸','감동이다']

# 긍정 1, 부정 0 라벨링
labels = np.array([1, 0, 0, 0,
                  1, 1, 1, 1,
                  1, 0, 0, 1, 
                  0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 1])


# tokenizer
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# {'돈': 1, '봐라': 2, '재밌어요': 3, '재미없다': 4, '아깝다': 5, '숙면했어요': 6, '최고예요': 7, 
#  '꼭': 8, '세9': 9, '번': 10, '또': 11, '보고싶다': 12, 'n회차': 13, '관람': 14, '배우가': 15, '잘': 16, 
#  '생기긴': 17, '했어요': 18, '발연기에요': 1, '추천해요': 20, '최악': 21, '후회된다': 22, '버렸다': 23,
#    '글쎄요': 24, '보다': 25, '나왔다': 26, '망작이다': 27, '연기가': 28, '어색해요': 29, '차라리': 30, 
#    '기부할걸': 31, '다음편': 32, '나왔으면': 33, '좋겠다': 34, '다른': 35, '거': 36, '볼걸': 37, '감동이다': 38}

x = token.texts_to_sequences(docs)
print(x)
# [[3], [4], [1, 5], [6], [7], [8, 2], [9, 10, 2], [11, 12], [13, 14], [15, 16, 17, 18], [19], [20], [21],
#  [22], [1, 23], [24], [25, 26], [27], [28, 29], [30, 31], [32, 33, 34], [35, 36, 37], [38]]

# pad_sequences
from keras_preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding = 'pre', maxlen = 4)   # 출력한 단어에 모두 0을 채움
print(pad_x)
print(pad_x.shape)  # (23, 4) --> 23은 전체 개수(x 크기, 라벨링 개수), 4는 maxlen

word_size = len(token.word_index)   # word_size :  38  --> 단어 사전 개수
print('word size : ',word_size)     # word size :  38 

# 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim = 39, #word_size : 38 + 1
                    # 각 단어의 인덱스는 0부터 시작하기 때문에, 모든 단어를 벡터화하기 위해서는 
                    # 0을 포함한 총 단어의 개수(word_size + 1)만큼의 차원이 필요
                    output_dim = 10,    # node 수 
                    input_length=4))    # 문장의 길이(가장 긴 길이에 맞춰짐)

model.add(LSTM(32)) # 시간적인 개념, 3차원을 받아서 2차원을 보냄
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 긍정, 부정 --> 이진분류

# model.summary()

# 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics = 'accuracy')

#model.fit(pad_x, labels, epochs=100, batch_size = 16 )
model.fit(pad_x, labels, epochs=10, batch_size = 1)
#4.평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)


# loss :  0.08198674768209457
# acc :  1.0

############################### [predict] ##################

# x_predict = '진짜 정말 후회된다 최악'
x_predict = '정말 정말 재미있고 최고예요'
#x_predict = "배우가 잘 생기긴 했네요"


#1) tokenizer
token = Tokenizer()
x_predict = np.array([x_predict])
print(x_predict)     # 리스트 안에 들어갔는지 확인
token.fit_on_texts(x_predict)   # fit on 하면서 index 가 생성됨
x_pred = token.texts_to_sequences(x_predict)
#print(token.word_index())
print(len(token.word_index))    # max length 맞춰줘야 함
print(x_pred)

score =float(model.predict(x_pred))


# 2) pad_sequences
x_pred = pad_sequences(x_pred, padding='pre')
print(x_pred)

# 3) model.predict
y_pred = model.predict(x_pred)
print(y_pred)
# [[0.9804912]]


# [실습]
# 1. predict 결과 제대로 출력되도록
# 2. score 를 긍정과 부정으로 출력하라!!

score = float(model.predict(x_pred))

if y_pred > 0.5:
    print("{:.2f}% 확률로 긍정\n".format(score * 100))
else:
    print("{:.2f}% 확률로 부정\n".format((1-score) * 100))
