import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:\AI_study\medical_noshow.csv')  

x = df.loc[:, df.columns != 'No-show']
y = df[['No-show']]   

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# 변수 설정
n_splits = 5   
random_state = 42
scaler = MinMaxScaler()


kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)


# Scaler
scaler.fit(x_train)                 
x_train = scaler.transform(x_train)   # train 은 fit, transform 모두 해줘야 함
x_test = scaler.transform(x_test) 

'''
#parameters_01
param = {
    'learning_rate': [0.1, 0.5, 1], # controls the learning rate
    'depth': [3, 4, 5], # controls the maximum depth of the tree
    'l2_leaf_reg': [2, 3, 4], # controls the L2 regularization term on weights
    'colsample_bylevel': [0.1, 0.2, 0.3], # specifies the fraction of columns to be randomly sampled for each level
    'n_estimators': [100, 200], # specifies the number of trees to be built
    'subsample': [0.1, 0.2, 0.3], # specifies the fraction of observations to be randomly sampled for each tree
    'border_count': [32, 64, 128],# specifies the number of splits for numerical features
    'bootstrap_type': ['Bernoulli', 'MVS']
} 

'''

#parameters_02
param = {
    'learning_rate': [0.1, 0.5, 1],
    'depth': [3, 4, 5],
    'l2_leaf_reg': [2, 3, 4],
    'colsample_bylevel': [0.1, 0.2, 0.3]
}


cat = CatBoostClassifier()
model = GridSearchCV(cat, param,  cv = kfold, 
                   refit = True, verbose = 1, n_jobs = -1  )

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train, early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)])
end_time = time.time() - start_time

print('최적의 파라미터 : ', model.best_params_ )
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')

# # 2. 모델
# model = CatBoostClassifier()

# # 3. 훈련
# model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)

score = cross_val_score( model, 
                        x_train, y_train,
                        cv = kfold )  #cv = cross validation_

y_predict = cross_val_predict(model,
                              x_test, y_test,
                              cv = kfold)

acc = accuracy_score(y_test, y_predict)

print('결과 acc : ', result)
print('cv pred acc : ', acc )

#결과 acc :  0.8005066497783407
# cv pred acc :  0.7988328960463222