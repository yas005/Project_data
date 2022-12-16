
##############################################################################
#########################주성분분석(PCA)######################################
##############################################################################

#공개 예제 데이터 가지고 오기.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print(cancer.data.shape)
print(cancer.target.shape)

#표준화(Nomalization)
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)

#주성분분해 과정.
import numpy as np
scov=np.cov(X_train_std.T)   #공분산 행렬구하기 
eigen_vals, eigen_vecs=np.linalg.eig(scov)  # 주성분 분해.
print('Eigenvalues \n%s' %eigen_vals)
print('Eigenvector \n%s' %eigen_vecs)

#설명비율 구하기.
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
#엘보우 차트그리기.
import matplotlib.pyplot as plt
plt.plot(cum_var_exp)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

plt.plot(var_exp)
plt.xlabel('number of components')
plt.ylabel('explained variance');




#sk-learn을 이용한 주성분 분석
from sklearn.decomposition import PCA
pca = PCA(n_components=2)    #주성분의 갯수
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std) #주성분 적용 변환
X_test_pca = pca.transform(X_test_std)  #주성분 적용 변환(test set 주의~~!!)
print("Original shape: {}".format(str(X_train_std.shape)))
print("Reduced shape: {}".format(str(X_train_pca.shape)))

#eigen value
print('eigen_value :', pca.explained_variance_)
print('explained variance ratio :', pca.explained_variance_ratio_)

#엘보우 그리기.
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');





#2개의 주성분만 가지고 그림을 그려보자.
import mglearn
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.legend(cancer.target_names, loc="best")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
#몬가 분류가 될듯하지 않나????

#로지스틱회귀 분석에 적용./
from sklearn.linear_model import LogisticRegression
lgogi_r=LogisticRegression()
#표준화된 자료
lgogi_r.fit(X_train_std, y_train)
print("훈련 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_train_std,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_test_std,y_test)))


#표준화 전 데이터
lgogi_r.fit(X_train, y_train)
print("훈련 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_test,y_test)))
#표준화 이후가 정확도가 높다.




#PCA의 적용
lgogi_r.fit(X_train_pca, y_train)
print("훈련 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_train_pca,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(lgogi_r.score(X_test_pca,y_test)))
#정확도 나쁘지 않음(30개 변수에서 2개의 변수로 축소)



#수고하셨습니다