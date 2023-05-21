ml25_joblib_load_cali.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# joblib 불러오기 // 2. 모델, 3. 훈련 
import joblib

path = './_data/'
model = joblib.load(path+'project01.dat')

#4. 평가, 예측
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
print('보팅 결과 : ', score)