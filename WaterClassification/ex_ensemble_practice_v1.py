##############################################################################
##################한국외대 빅데이터 비지니스 응용 및 실습#######################
##############################################################################
########################앙상블 학습###########################################
##############################################################################

#실습용 데이터 만들기
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.4, random_state=0)
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1],c=y)
plt.show()

#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


#여러가지 모형의 적합.
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
log_clf = LogisticRegression(random_state=42)
svm_clf = SVC(random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)

#하드 보팅  VS 소프트 보팅
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('tree', tree_clf), ('svc', svm_clf)],
    voting='hard')                           #soft voting의 경우 voting='soft'
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, tree_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



#단일 트리의 결과.
tree_clf.fit(X_train, y_train)
y_pred_tree2 = tree_clf.predict(X_train)
y_pred_tree = tree_clf.predict(X_test)
print('Train Accuracy =', accuracy_score(y_train, y_pred_tree2))
print('Test Accuracy =', accuracy_score(y_test, y_pred_tree))


#배깅을 사용한 tree 모형.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), 
    n_estimators=500,   #배깅의 횟수
    max_samples=100,    #1회 샘플시 max 샘플 사이즈 
    bootstrap=True,     #복원추출여부 (boostrap)
    n_jobs=-1,          #가용 CPU '-1' 전부.
    random_state=42)

bag_clf.fit(X_train, y_train)
y_pred_train=bag_clf.predict(X_train)   #test data 사용하여 평가하기.
y_pred=bag_clf.predict(X_test)   #test data 사용하여 평가하기.
print('Train Accuracy =', accuracy_score(y_train, y_pred_train))
print('Test Accuracy =', accuracy_score(y_test, y_pred))

###결정경계면 그림 확인하기
import numpy as np
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5,
                           contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Tree by Bagging", fontsize=14)
plt.show()



#OOB data사용하여 평가하기.
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
print('oob score :', bag_clf.oob_score_)




###########랜덤 포레스트 VS  tree 배깅#########################
from sklearn.ensemble import RandomForestClassifier
# BaggingClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, 
                           random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, 
    random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test) #test data 사용하여 평가하기.
print('Bagging tree Accuracy =', accuracy_score(y_test, y_pred))


# Random Forest
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test) #test data 사용하여 평가하기.
print('random forest Accuracy =', accuracy_score(y_test, y_pred_rf))


plt.figure(figsize=(6, 4))
for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], 
                           alpha=0.02, contour=False)
plt.show()



###########부스팅#########################
#adaboost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
ada_clf.score(X_test, y_test) 
plot_decision_boundary(ada_clf, X, y)


#gradiant boosting
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X_train,y_train)

print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test,y_test)))
plot_decision_boundary(gbrt, X, y)

#수고하셨습니다