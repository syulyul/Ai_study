from keras.preprocessing.text import Tokenizer

# text = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 많이 많이 많이 먹었다'
text = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 엄청 많이 많이 많이 먹어서 매우 배가 부르다'

token = Tokenizer()
token.fit_on_texts([text])  # fit on 하면서 index 가 생성됨

# index = token.word_index
print(token.word_index)
# {'매우': 1, '많이': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# {'매우': 1, '많이': 2, '엄청': 3, '나는': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '먹어서': 8, '배가': 9, '부
# 르다': 10}

x = token.texts_to_sequences([text])
print(x)


from keras.utils import to_categorical

x = to_categorical(x)   # onehotencoding 하면 index 수 + 1개로 만들어짐
print(x)    # 11 개 인 이유 : 빈칸을 만들어줌 --> 훈련 데이터에 없는 경우를 처리해 주기 위함
print(x.shape)  # (1, 17, 11)   --> 3차원으로 받아온다

'''
########## OneHotEncoder 수정 ##########
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x = x.reshape(-1, 11, 9)
onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape)
# AttributeError: 'list' object has no attribute 'reshape'
'''