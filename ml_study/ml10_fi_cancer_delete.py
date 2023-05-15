import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_breast_cancer()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# drop_features
# x = np.delete(x, 0, axis=1)
# cv pred acc :  0.956140350877193
# x = np.delete(x, 1, axis=1)
# x = np.delete(x, [0, 1], axis=1)
# cv pred acc :  0.9473684210526315
x = np.delete(x, [0, 1, 2], axis=1)

# kfold
n_splits = 11    # 보통 홀수로 들어감
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)                 # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_train) # train 은 fit, transform 모두 해줘야 함
x = scaler.transform(x_test) 



# 2. 모델
model = RandomForestClassifier()
# model = DecisionTreeClassifier()

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
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9333333333333333
# stratifiedKFold cv pred acc :  0.9333333333333333


###### feature importance ##########
print(model, " : ", model.feature_importances_)

# cv pred acc :  0.9333333333333333
# RandomForestClassifier()  :  [0.07901659 0.0336995  0.44851384 0.43877007]

# 시각화
import matplotlib.pyplot as plt
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('cancer Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)

plt.show()

# cv pred acc :  0.9473684210526315
# RandomForestClassifier()  :  [0.04388963 0.01241847 0.03360554 0.06143427 0.00450459 0.00570493
#  0.04031094 0.15657512 0.00321955 0.0042518  0.01034653 0.00492772
#  0.00847911 0.04408195 0.00490235 0.00583323 0.00478525 0.00386004
#  0.00415095 0.00610938 0.09887963 0.02108879 0.10951271 0.08503506
#  0.01238128 0.01323798 0.03523214 0.13963324 0.01194079 0.00966702]