import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_breast_cancer()  # 다중분류
x = datasets['data']
y = datasets.target

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42

)
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
from xgboost import XGBClassifier
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error')  # 이진분류 : error
# cv pred acc :  0.9473684210526315


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


# SelectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   # prefit=True --> 자신보다 작은 걸 가지고온다
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape)
    print(select_x_test.shape)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, acc:%.2f%%"
        %(thresh, select_x_train.shape[1], score*100))
    

# cv pred acc :  0.9473684210526315
# (455, 15)
# (114, 15)
# Thresh=0.008, n=15, acc:95.61%
# (455, 8)
# (114, 8)
# Thresh=0.024, n=8, acc:95.61%
# (455, 13)
# (114, 13)
# Thresh=0.014, n=13, acc:95.61%
# (455, 9)
# (114, 9)
# Thresh=0.020, n=9, acc:95.61%
# (455, 16)
# (114, 16)
# Thresh=0.006, n=16, acc:96.49%
# (455, 18)
# (114, 18)
# Thresh=0.005, n=18, acc:96.49%
# (455, 7)
# (114, 7)
# Thresh=0.028, n=7, acc:94.74%
# (455, 1)
# (114, 1)
# Thresh=0.401, n=1, acc:87.72%
# (455, 27)
# (114, 27)
# Thresh=0.002, n=27, acc:96.49%
# (455, 21)
# (114, 21)
# Thresh=0.004, n=21, acc:96.49%
# (455, 14)
# (114, 14)
# Thresh=0.013, n=14, acc:95.61%
# (455, 19)
# (114, 19)
# Thresh=0.005, n=19, acc:96.49%
# (455, 10)
# (114, 10)
# Thresh=0.018, n=10, acc:95.61%
# (455, 20)
# (114, 20)
# Thresh=0.004, n=20, acc:96.49%
# (455, 22)
# (114, 22)
# Thresh=0.004, n=22, acc:96.49%
# (455, 23)
# (114, 23)
# Thresh=0.004, n=23, acc:96.49%
# (455, 6)
# (114, 6)
# Thresh=0.029, n=6, acc:94.74%
# (455, 28)
# (114, 28)
# Thresh=0.002, n=28, acc:95.61%
# (455, 29)
# (114, 29)
# Thresh=0.001, n=29, acc:95.61%
# (455, 25)
# (114, 25)
# Thresh=0.002, n=25, acc:96.49%
# (455, 5)
# (114, 5)
# Thresh=0.055, n=5, acc:95.61%
# (455, 11)
# (114, 11)
# Thresh=0.016, n=11, acc:95.61%
# (455, 4)
# (114, 4)
# Thresh=0.057, n=4, acc:95.61%
# (455, 3)
# (114, 3)
# Thresh=0.067, n=3, acc:95.61%
# (455, 17)
# (114, 17)
# Thresh=0.005, n=17, acc:96.49%
# (455, 26)
# (114, 26)
# Thresh=0.002, n=26, acc:96.49%
# (455, 12)
# (114, 12)
# Thresh=0.015, n=12, acc:95.61%
# (455, 2)
# (114, 2)
# Thresh=0.184, n=2, acc:87.72%
# (455, 24)
# (114, 24)
# Thresh=0.003, n=24, acc:96.49%
# (455, 30)
# (114, 30)
# Thresh=0.000, n=30, acc:95.61%