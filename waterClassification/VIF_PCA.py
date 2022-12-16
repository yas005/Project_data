import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

data=pd.read_csv('D:/데이터 마이닝/water_potability.csv', encoding='CP949')
data.head()

x=data[["ph","Hardness","Solids","Chloramines","Conductivity",
        "Organic_carbon","Trihalomethanes","Turbidity"]]
y=data[["Potability"]]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)

#%% vif 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
vif["features"] = data.columns
vif

#%%상관행렬 확인하기
# 상관행렬 보기
data.corr()
# 상관행렬 시각화
import matplotlib.pyplot as plt
import seaborn as sns  #heatmap 만들기 위한 라이브러리
cmap = sns.light_palette("darkgray", as_cmap = True)  
sns.heatmap(data.corr(), annot = True, cmap = cmap)
plt.show()

#%%PCA 분석
#수치형 변수 정규화
from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

#최적의 주성분 개수 구하기
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(x)
pc_score=pca.transform(x) #pc값이 클수록 설명력이 높다는 의미
pca.components_
pca.explained_variance_
ratio=pca.explained_variance_ratio_
ratio
#설명력 정도 확인
df_v=pd.DataFrame(ratio, index=['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6', 'PC7','PC8'], columns=['pca_ratio'])
df_v.plot.pie(y='pca_ratio')
df_v

pca=PCA(n_components=8) #PCA 객체 생성(주성분 객수 8개 생성)
principalComponents=pca.fit_transform(x)
principalDf=pd.DataFrame(data=principalComponents,columns=['주성분1', '주성분2','주성분3','주성분4','주성분5','주성분6','주성분7','주성분8'])
principalDf #주성분 점수 확인

#최적의 주성분 개수로 분석하기
import numpy as np
pca=PCA(n_components=5)
pc=pca.fit_transform(x)
pc_y=np.c_[pc,y]
principalDf=pd.DataFrame(pc_y, columns=['주성분1','주성분2','주성분3','주성분4', '주성분5', 'diagnosis'])
principalDf

#%% PCA 처리한 데이터에 로지스틱 회귀분석 진행
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#로지스틱 회귀분석 모델 생성, 다중 분류이므로 multi_class를 적용한다.
clf=LogisticRegression(max_iter=1000, random_state=0, multi_class='multinomial')
clf.fit(x[:,:2],y)
pred=clf.predict(x[:,:2])
confusion_matrix(y,pred)