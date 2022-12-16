
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm


data=pd.read_csv('D:/데이터 마이닝/water_potability.csv')
data.head()

x=data[["ph","Hardness","Solids","Chloramines","Conductivity",
        "Organic_carbon","Trihalomethanes","Turbidity"]]
y=data[["Potability"]]

# 훈련 vs 테스트의 분리(70:30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=0
                                                    # sampling 초기치
                                                    )
print("training X Data shape: {}".format(x_train.shape))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression
# 라이브러리 세팅.
LR=LogisticRegression(solver='liblinear',multi_class='auto' , random_state=1) 
#모형의 적합
LR.fit(x_train,y_train)


#train set에 대한 정확도
print(LR.score(x_train, y_train))
#test set에 대한 정확도
print(LR.score(x_test, y_test))
#각 변수들의 계수를 확인>> 어떤 변수가 가장 많은 영향을 주는지 확인하기 위함
print(LR.coef_)

y_train_pred=LR.predict(x_train)       # train data set에 대한 y의 예측치.
y_test_pred=LR.predict(x_test)        # test set에 대한 y의 예측치.

#오분류표
conf1=confusion_matrix(y_true=y_train,y_pred=y_train_pred)
print(conf1)  # 훈련데이터 오분류표
conf2=confusion_matrix(y_true=y_test,y_pred=y_test_pred) 
print(conf2) # 테스트 데이터 오분류표

y_prob_train=LR.predict_proba(x_train) # test set 기준 관측치별 1이 될 확률.
y_prob_test=LR.predict_proba(x_test) # test set 기준 관측치별 1이 될 확률. 

#ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,y_prob_test[:, 1])
auc(false_positive_rate, true_positive_rate)
roc_auc_score(y_test, clf.predict(x_test))
