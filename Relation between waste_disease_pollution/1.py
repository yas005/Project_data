# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:08:21 2021

@author: user
"""
import pandas as pd 
import numpy as np
import statsmodels.api as sm

data = pd.read_csv(r'F:/2021/2021년 1학기/데이터 프로세싱/data_2010_2019_fin.csv', encoding='cp949')
print(data)

x_data = data[["combustible waste", "ozone", "sulfur dioxide",
               "nitrogen dioxide", "carbon monoxide"]] 
target1= data[["parkinson's disease"]]
target2= data[["asthma"]]
target3= data[["chronic bronchitis"]]
target4= data[["angina pectoris"]]

x_data1 = sm.add_constant(x_data, has_constant="add")

multi_model_1 = sm.OLS(target1, x_data1)
fitted_multi_model_1 = multi_model_1.fit()
fitted_multi_model_1.summary()

multi_model_2 = sm.OLS(target2, x_data1)
fitted_multi_model_2 = multi_model_2.fit()
fitted_multi_model_2.summary()

multi_model_3 = sm.OLS(target3, x_data1)
fitted_multi_model_3 = multi_model_3.fit()
fitted_multi_model_3.summary()

multi_model_4 = sm.OLS(target4, x_data1)
fitted_multi_model_4 = multi_model_4.fit()
fitted_multi_model_4.summary()

x_data2 = data[["combustible waste", "ozone", "sulfur dioxide",
               "nitrogen dioxide", "carbon monoxide"]]
x_data2.head()
x_data2= sm.add_constant(x_data2, has_constant = "add")
multi_model2 = sm.OLS(target1, x_data2)
fitted_multi_model2 = multi_model2.fit()
fitted_multi_model2.summary()

multi_model2 = sm.OLS(target2, x_data2)
fitted_multi_model2 = multi_model2.fit()
fitted_multi_model2.summary()

multi_model2 = sm.OLS(target3, x_data2)
fitted_multi_model2 = multi_model2.fit()
fitted_multi_model2.summary()

multi_model2 = sm.OLS(target4, x_data2)
fitted_multi_model2 = multi_model2.fit()
fitted_multi_model2.summary()


from sklearn.model_selection import train_test_split
X = x_data2_
y = target
train_x, test_x, train_y, test_y = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 1)
# 학습데이터와 검증데이터를 7:3으로 분리한다.
# random_state고정을 통해 그때마다 똑같은 값을 분류하도록 한다.

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
# train_x에 상수항 추가 후 최귀모델 적합

fit_train1 = sm.OLS(train_y,train_x)
fit_train1 = fit_train1.fit()

# 검증데이터에 대한 예측값과 true값 비교

plt.plot(np.array(fit_train1.predict(test_x)),label = "pred")
plt.plot(np.array(test_y),label = "true")
plt.legend()
plt.show()
# x_data3와 x_data4 학습 검증데이터 분할

X = x_data3_
y = target
train_x2,test_x2,train_y2,test_y2 = train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=1)

X = x_data4_
y = target
train_x3,test_x3,train_y3,test_y3 = train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=1)

# x_data3/x_data4의 회귀모델 적합 (fit_train2,fit_train3)

fit_train2 = sm.OLS(train_y2,train_x2)
fit_train2 = fit_train2.fit()

fit_train3 = sm.OLS(train_y3,train_x3)
fit_train3 = fit_train3.fit()

# vif를 통해 NOX를 지운 데이터 x_data3 , NOX,RM을 지운 데이터 x_data4 full모델 실제값 비교

plt.plot(np.array(fit_train1.predict(test_x)),label = "pred_full")
plt.plot(np.array(fit_train2.predict(test_x2)),label = "pred_vif")
plt.plot(np.array(fit_train3.predict(test_x3)),label = "pred_vif2")
plt.plot(np.array(test_y2), label = "true")
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error

#변수 제거가 이루어지지 않은 full모델
mse1 = mean_squared_error(y_true = test_y["Target"], y_pred = fit_train1.predict(test_x))

#변수 NOX만 제거한 모델
mse2 = mean_squared_error(y_true = test_y["Target"], y_pred = fit_train2.predict(test_x2))

#변수 NOX와 RM 두 개를 제거한 모델
mse3 = mean_squared_error(y_true = test_y["Target"], y_pred = fit_train3.predict(test_x3))

print(mse1)
print(mse2)
print(mse3)
`   ```             