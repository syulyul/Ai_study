from keras.preprocessing.text import Tokenizer

# text = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 많이 많이 많이 먹었다'
text1 = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 엄청 많이 많이 많이 먹어서 매우 배가 부르다'
text2 = '나는 딥러닝이 정말 재미있다. 재미있어하는 내가 너무 너무 너무 너무 멋있다. 또 또 또 얘기해 봐.'

token = Tokenizer()
token.fit_on_texts([text1, text2])  # fit on 하면서 index 가 생성됨

print(token.word_index)
# {'매우': 1, '너무': 2, '많이': 3, '또': 4, '나는': 5, '엄청': 6, '진짜': 7, 
#  '맛있는': 8, '밥을': 9, '먹어서': 10, '배가': 11, '부르다': 12, '딥러닝이': 13, 
#  '정말': 14, '재미있다': 15, '재미있어하는': 16, '내가': 17, '멋있다': 18, 
#  '얘기해': 19, '봐': 20}

x = token.texts_to_sequences([text1, text2])
print(x)
# [[5, 7, 1, 1, 1, 1, 8, 9, 6, 6, 3, 3, 3, 10, 1, 11, 12],  # text1
#  [5, 13, 14, 15, 16, 17, 2, 2, 2, 2, 18, 4, 4, 4, 19, 20]]    # text


from keras.utils import to_categorical
x_new = x[0] + x[1]
print(x_new)    # 리스트 2개가 연결됨
'''
x = to_categorical(x_new)   # onehotencoding 하면 index 수 + 1개로 만들어짐
print(x)   
print(x.shape)  # (33, 21)
''' 


########## OneHotEncoder 수정 ##########
import numpy as np
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(
    categories='auto', sparse=False
    )

x = np.array(x_new)     # 배열 만들어주는 거
# x = x.reshape(-1, 11, 9)    # reshape : list로 해야됨
print(x.shape)  # (33,)
x = x.reshape(-1, 1)    # -1 : 차원 늘어남
print(x.shape)  # (33, 1)

onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape)  # (33, 20)  --> 0부터 시작하기 때문에 to_categorical 보다 1 큼

# AttributeError: 'list' object has no attribute 'reshape'