from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

# 스케일링
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer()                     # QuantileTransformer 는 지정된 분위수에 맞게 데이터를 변환함. 
                                                # 기본 분위수는 1,000개이며, n_quantiles 매개변수에서 변경할 수 있음
ptf1 = PowerTransformer(method='yeo-johnson')   # 'yeo-johnson', 양수 및 음수 값으로 작동
ptf2 = PowerTransformer(method='box-cox')       # 'box-cox', 양수 값에서만 작동

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )
    

#=================================== 결과 =====================================#
# StandardScaler 결과 : 1.0000
# MinMaxScaler 결과 : 1.0000
# MaxAbsScaler 결과 : 1.0000
# RobustScaler 결과 : 1.0000
# QuantileTransformer 결과 : 1.0000
# PowerTransformer 결과 : 1.0000
# ValueError: The Box-Cox transformation can only be applied to strictly positive data
#==============================================================================#