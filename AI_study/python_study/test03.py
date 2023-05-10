import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("Original(안바꾼거) :\n", a)     # \n --> 줄바꿈, \t --> 띄어쓰기

a_transpose = np.transpose(a)           # 행렬을 바꿔주는 함수
print("Transpose(바꾼거) :\n", a_transpose)
#  [[1 4]
#  [2 5]
#  [3 6]]

a_reshape = np.reshape(a, (3,2))        # 행렬을 바꿔주는 함수
print("Reshape(바꾼거) :\n", a_reshape)
#  [[1 2]
#  [3 4]
#  [5 6]]

# transpose() 와 reshape() 의 차이점 :
# transpose() --> 데이터의 순서대로 바뀜
# reshape()  --> 데이터를 변형해서 바뀜