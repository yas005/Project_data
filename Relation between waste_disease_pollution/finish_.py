## 다중회귀분석

import pandas as pd 
import numpy as np
import statsmodels.api as sm

#데이터 불러오기
data=r'F:\2021\2021년 1학기\데이터 프로세싱/data_2010_2019_fin.csv'
finish_data = pd.read_csv(data)

#다중선형회귀분석(데이터 추출)
y_data = finish_data.drop(["parkinson's disease"], axis=1)
x_data = finish_data[["combustible waste","ozone","sulfur dioxide","nitrogen dioxide"]]
y = finish_data[["asthma"]]
df = x_data.drop(columns="nitrogen dioxide")

# 상수항 추가
x_data_ = sm.add_constant(df, has_constant = "add")

# 회귀모델 적합
multi_model2 = sm.OLS(y, x_data_)
fitted_multi_model2 = multi_model2.fit()

# 결과 출력
fitted_multi_model2.summary()

# 상관행렬 보기
#df.corr()

# 상관행렬 시각화
#import matplotlib.pyplot as plt
#import seaborn as sns  #heatmap 만들기 위한 라이브러리
#cmap = sns.light_palette("darkgray", as_cmap = True)  
#sns.heatmap(df.corr(), annot = True, cmap = cmap)
#plt.show()

#sns.pairplot(df)
#plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
vif["features"] = df.columns
vif

##크롤링
import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

date = str(datetime.now())
date = date[:date.rfind(':')].replace(' ', '_')
date = date.replace(':','시') + '분'

query = input('검색 키워드를 입력하세요 : ')
news_num = int(input('총 필요한 뉴스기사 수를 입력해주세요(숫자만 입력) : '))
query = query.replace(' ', '+')


news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}'

req = requests.get(news_url.format(query))
soup = BeautifulSoup(req.text, 'html.parser')


news_dict = {}
idx = 0
cur_page = 1

print()
print('크롤링 중...')

while idx < news_num:
### 네이버 뉴스 웹페이지 구성이 바뀌어 태그명, class 속성 값 등을 수정함(20210126) ###
    
    table = soup.find('ul',{'class' : 'list_news'})
    li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
    area_list = [li.find('div', {'class' : 'news_area'}) for li in li_list]
    a_list = [area.find('a', {'class' : 'news_tit'}) for area in area_list]
    
    for n in a_list[:min(len(a_list), news_num-idx)]:
        news_dict[idx] = {'title' : n.get('title'),
                          'url' : n.get('href') }
        idx += 1

    cur_page += 1

    pages = soup.find('div', {'class' : 'sc_page_inner'})
    next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
    
    req = requests.get('https://search.naver.com/search.naver' + next_page_url)
    soup = BeautifulSoup(req.text, 'html.parser')

print('크롤링 완료')

print('데이터프레임 변환')
news_df = DataFrame(news_dict).T

folder_path = os.getcwd()
xlsx_file_name = '네이버뉴스_{}_{}.xlsx'.format(query, date)

news_df.to_excel(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}'.format(folder_path, xlsx_file_name))
os.startfile(folder_path)

