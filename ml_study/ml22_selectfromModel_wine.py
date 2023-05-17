import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77)  # weight 의 난수값 조정

# 1. 데이터
datasets = load_wine()  # 다중분류
x = datasets['data']
y = datasets.target
feature_name = datasets.feature_names
print(feature_name)

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
          eval_metric='merror')  # 다중분류 : merror
# cv pred acc :  0.8611111111111112


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
    
# 컬럼명 출력
selected_feature_indices = selection.get_support(indices=True)
selected_feature_names = [feature_name[i] for i in selected_feature_indices]
print(selected_feature_names)

# cv pred acc :  0.8611111111111112
# (142, 7)
# (36, 7)
# Thresh=0.014, n=7, acc:94.44%
# (142, 10)
# (36, 10)
# Thresh=0.009, n=10, acc:97.22%
# (142, 8)
# (36, 8)
# Thresh=0.013, n=8, acc:94.44%
# (142, 11)
# (36, 11)
# Thresh=0.005, n=11, acc:97.22%
# (142, 5)
# (36, 5)
# Thresh=0.031, n=5, acc:94.44%
# (142, 6)
# (36, 6)
# Thresh=0.017, n=6, acc:94.44%
# (142, 4)
# (36, 4)
# Thresh=0.119, n=4, acc:94.44%
# (142, 13)
# (36, 13)
# Thresh=0.000, n=13, acc:97.22%
# (142, 12)
# (36, 12)
# Thresh=0.001, n=12, acc:97.22%
# (142, 2)
# (36, 2)
# Thresh=0.173, n=2, acc:88.89%
# (142, 9)
# (36, 9)
# Thresh=0.011, n=9, acc:94.44%
# (142, 1)
# (36, 1)
# Thresh=0.469, n=1, acc:61.11%
# (142, 3)
# (36, 3)
# Thresh=0.139, n=3, acc:91.67%