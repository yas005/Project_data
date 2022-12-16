##############################################################################
#########################표준화 (Nomalization)##################################
##############################################################################
import pandas as pd

#공개 예제 데이터 가지고 오기.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

#데이터의 개요.
print(cancer.DESCR)
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
sy = pd.Series(cancer.target, dtype="category")
sy = sy.cat.rename_categories(cancer.target_names)
df['class'] = sy
import seaborn as sns
from matplotlib import pyplot as plt
sns.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
             hue="class", data=df)
plt.show()



#훈련데이터와 테스트 데이터의 분리
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print(X_train.shape)
print(X_test.shape)



#Minmax 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

#train data의 변환
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling
print("변화후 데이터 모양: {}".format(X_train_scaled.shape))
print("변환전 최소값:\n {}".format(X_train.min(axis=0)))
print("변환전 최대값:\n {}".format(X_train.max(axis=0)))
print("변환후 최소값:\n {}".format(X_train_scaled.min(axis=0)))
print("변환후 최대값:\n {}".format(X_train_scaled.max(axis=0)))

#test set의 변환.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled = test_scaler.transform(X_test)
print("변환전 최소값:\n {}".format(X_test.min(axis=0)))
print("변환전 최대값:\n {}".format(X_test.max(axis=0)))
print("변환후 최소값(test set):\n{}".format(X_test_scaled.min(axis=0)))
print("변환후 최대값(test set):\n{}".format(X_test_scaled.max(axis=0)))


#%% 올바른 test set의 변환.
X_test_scaled_good = scaler.transform(X_test)
# print test data properties after scaling
print("변환후 최소값(test set):\n{}".format(X_test_scaled_good.min(axis=0)))
print("변환후 최대값(test set):\n{}".format(X_test_scaled_good.max(axis=0)))

df2 = pd.DataFrame(X_test_scaled_good,columns=cancer.feature_names)
df2['class'] = sy
sns.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
             hue="class", data=df2)
plt.show()

#####표준화의 중요성.
from sklearn.svm import SVC
svm = SVC(C=100,gamma='auto')
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

#표준화된 데이터의 사용.
svm.fit(X_train_scaled, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test_scaled_good, y_test)))


#수고하셨습니다.