##############################################################################
#########################SVM(써포트 벡터머신)##################################
##############################################################################

#%%##########Step0. 필요한 모듈 불러오기 ##########
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#%%##########Step0-1. 필요한 함수 정의하기 ##########
# def 2개는 복사하여 붙여놓고 실행 시키키 바람.
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


#%%####################Step1. 예제 data의 생성 ##################################
#군집화된 데이터 셋의 생성.
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, n_features=2, centers=2, \
                  cluster_std=2.0,random_state=3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()
    

#%%###############Step2. trainning vs test########## ##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                            test_size=0.25,random_state=8)
print("test X Data shape: {}".format(X_test.shape))
print("training X Data shape: {}".format(X_train.shape))















#%%##########Step3. 모형의 추정(SVM) ##########################
#선형 SVM 
from sklearn.svm import LinearSVC
#c=1
l_svc1=LinearSVC()
l_svc1.fit(X_train,y_train)
print("training set: {:.2f}".format(l_svc1.score(X_train, y_train)))
print("test set: {:.2f}".format(l_svc1.score(X_test, y_test)))


#c=10
l_svc01=LinearSVC(C=10000, max_iter=100000)
l_svc01.fit(X_train,y_train)
print("training set: {:.2f}".format(l_svc01.score(X_train, y_train)))
print("test set: {:.2f}".format(l_svc01.score(X_test, y_test)))
#c=0.01
l_svc10=LinearSVC(C=0.00001, max_iter=1000000)
l_svc10.fit(X_train,y_train)
print("training set: {:.2f}".format(l_svc10.score(X_train, y_train)))
print("test set: {:.2f}".format(l_svc10.score(X_test, y_test)))

#chart 그리기
plot_predictions(l_svc1, [-15, 5, -10, 10])
plot_dataset(X_train, y_train, [-15, 5, -10, 10])
plt.show()
#chart 그리기
plot_predictions(l_svc01, [-15, 5, -10, 10])
plot_dataset(X_train, y_train, [-15, 5, -10, 10])
plt.show()
#chart 그리기
plot_predictions(l_svc10, [-15, 5, -10, 10])
plot_dataset(X_train, y_train, [-15, 5, -10, 10])
plt.show()























#%%#########################credit data의 활용#################################
#전처리 완료된 데이터 가지고 오기
path=r'C:\Users\YB_Notebook\Documents\cs-training_f.csv'
training_data = pd.read_csv(path)
training_data = training_data.drop('Unnamed: 0', axis = 1)

print("Data shape: {}".format(training_data.shape))
training_data.columns[0]
training_data.columns[1:]    #특성변수의 확인

X2 = training_data.drop('seriousdlqin2yrs', axis=1)   #특성변수
y2 = training_data.seriousdlqin2yrs                   #타겟변수

#data가 svm추정에 너무커서 1%만 추출
X_train3, X_use, y_train3, y_use = train_test_split(X2, y2, \
                                            test_size=0.01,random_state=8)
print("test X Data shape: {}".format(X_use.shape))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_use, y_use, \
                                            test_size=0.25,random_state=8)
print("test X Data shape: {}".format(X_test2.shape))
print("training X Data shape: {}".format(X_train2.shape))

#credit data의 SVM
credit_svc=LinearSVC(C=0.0001,max_iter=10000000)
credit_svc.fit(X_train2,y_train2)
print("training set: {:.2f}".format(credit_svc.score(X_train2, y_train2)))
print("test set: {:.2f}".format(credit_svc.score(X_test2, y_test2)))

#특성변수의 분포 살펴보기
pd.DataFrame(X_train2).boxplot()

#정규화의 문제.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train2) #정규화실행
X_train_std=sc.transform(X_train2)
X_test_std=sc.transform(X_test2)
#정규화된 data 분포 살펴보기
X_train_std.boxplot()
#정규화된 data그림 그리기 위한 변환작업
df_X_train_std=pd.DataFrame(X_train_std)
df_X_train_std.boxplot()


credit_svc_std=LinearSVC(max_iter=10000)
credit_svc_std.fit(X_train_std,y_train2)
print("traininset: {:.2f}".format(credit_svc_std.score(X_train_std, y_train2)))
print("test set: {:.2f}".format(credit_svc_std.score(X_test_std, y_test2)))

# 슬랙변수의 도입
from sklearn.svm import LinearSVC
slack_1=LinearSVC(C=1.0,max_iter=10000).fit(X_train_std,y_train2)
print("training set: {:.2f}".format(slack_1.score(X_train_std, y_train2)))
print("test set: {:.2f}".format(slack_1.score(X_test_std, y_test2)))

slack_01=LinearSVC(C=0.01,max_iter=10000).fit(X_train_std,y_train2)
print("training set: {:.2f}".format(slack_01.score(X_train_std, y_train2)))
print("test set: {:.2f}".format(slack_01.score(X_test_std, y_test2)))

slack_100=LinearSVC(C=100,max_iter=1000000).fit(X_train_std,y_train2)
print("training set: {:.2f}".format(slack_100.score(X_train_std, y_train2)))
print("test set: {:.2f}".format(slack_100.score(X_test_std, y_test2)))

























































#%%##########################Kernel SVM #####################################
# 예제 데이터의 생성
from sklearn.datasets import make_blobs
import mglearn
#그림 그리기 라이브러리.
# Introduction to Machine Learning with Python by Andreas Muller
# pip install mglearn  이용하여 설치
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, \
                  cluster_std=1.0, random_state=8)
y=y%2  #data를 '0'과 '1'로 만들기 위한.
#mglearn library사용항 chart 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


#선형 SVM 적용하여 보기
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


##특성 중 일부의 제곱항(생성)이용하여 3D Plot그리기.
X_new = np.hstack([X, X[:, 1:] ** 2]) #두번재 특성의 제곱항 추가


from mpl_toolkits.mplot3d import Axes3D #for 3D plot
figure = plt.figure()
# 3D plot setting
ax = Axes3D(figure, elev=-152, azim=-26)
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', \
           marker='^',cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()



#커널 SVM의 적합
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# show linear decision boundary
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',\
           marker='^',cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature0 ** 2")
plt.show()



#원래 평면상에 결정선 확인하기.
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()





#%%SVC library의 활용.
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plot support vectors
sv = svm.support_vectors_
# class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()





#c vs gamma example
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
ncol=4, loc=(.9, 1.2))


# 하이퍼 파라메터 C와 gamma를 바꾸어 가면서 최적 모형을 탐색








#문제 1 : 위 예제의 데이터를 training data 75%, test data 25%로 나누어 
#         커널 SVM (kernel='rbf')를 적용하여 train set과 test set의 분류 정확도를
#         구하시오.
#         단. data의 'cluster_std=2.5'등 바꾸어 가며 작업하시오.

#문제 2 : 위문제의 결정 경계선을 그림으로 그리시오.

#문제 3 : KNN과 tree모형으로 적합하여 결과를 비교 하시오.



#수고 하셨습니다.