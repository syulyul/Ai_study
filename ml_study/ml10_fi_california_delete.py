import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = fetch_california_housing()  # 다중분류
x = datasets['data']
y = datasets.target
print(datasets.DESCR)

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# drop_features
# x = np.delete(x, 0, axis=1)
# cv pred acc :  0.956140350877193
# x = np.delete(x, 1, axis=1)
# x = np.delete(x, [0, 1], axis=1)
# cv pred acc :  0.9473684210526315

# kfold
n_splits = 11    # 보통 홀수로 들어감
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_train) # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_test) 



# 2. 모델
model = RandomForestRegressor()
# model = DecisionTreeRegressor()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = cross_val_score(model, 
                        x_train, y_train, 
                        cv=kfold)   # cv : corss validation
# print('cv acc : ', score)   # kfold 에 있는 n_splits 숫자만큼 나옴 
y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv=kfold)
# print('cv pred : ', y_predict)
# cv pred :  [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 1 0 2 2 2 2 2 0 0] # 0.8% 를 제외한 나머지 0.2 %
r2 = r2_score(y_test, y_predict)
print('cv pred acc : ', r2)


###### feature importance ##########
print(model, " : ", model.feature_importances_)

# cv pred acc :  0.9333333333333333
# RandomForestClassifier()  :  [0.07901659 0.0336995  0.44851384 0.43877007]

# 시각화
import matplotlib.pyplot as plt
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('california Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)

plt.show()

# cv pred acc :  0.7470305152982859
# RandomForestRegressor()  :  [0.52610602 0.05422273 0.0450391  0.02972732 0.03084186 0.1379297
#  0.08836176 0.08777151]