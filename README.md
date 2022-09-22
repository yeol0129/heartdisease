# 머신러닝 심장질환 예측

#### 자세한 코드는 [여기있습니다](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb)
## Data
> ### heart.csv의 데이터 예
> Age|Sex|ChestPainType|RestingBP|Cholesterol|FastingBS|RestingECG|MaxHR|ExerciseAngina|Oldpeak|ST_Slope|HeartDisease
> ---|---|---|---|---|---|---|---|---|---|---|---|
> 40|M|ATA|140|289|0|Normal|172|N|0|Up|0
> 49|F|NAP|160|180|0|Normal|156|N|1|Flat|1
> 37|M|ATA|130|283|0|ST|98|N|0|Up|0
> ```
> Age : 나이 
> Sex : 성별(0 : 여성; 1 : 남성) 
> CPT : 가슴통증 유형(무증상(ASY) ; 이례적 협심증(ATA); 비협심증 통증(NAP);  일반적 협심증(TA))
> RestingBP : 평상시 혈압 
> Cholesterol : 혈중 콜레스테롤 농도
> FastingBS : 공복시 혈당 > 120mg/dl(0 = False; 1 = True)
> RestingECG : 안정 심전도 결과(좌심실 비대(LVH);  정상(Normal);  심전도 비정상(ST))
> MaxHR : 최대 심박수 
> ExerciseAngina : 협심증 유발 운동
> Oldpeak : 비교적 안정되기까지 운동으로 유발되는 심전도
> ST_Slope : 최대 운동 심전도의 기울기
> HeartDisease : 심장병 진단(0 = False; 1 = True)
> ```

## 사용 라이브러리
> ```python
> from keras.models import Sequential
> from keras.layers import Dense
> from sklearn.model_selection import train_test_split
> from sklearn.preprocessing import LabelEncoder
> from keras.callbacks import ModelCheckpoint, EarlyStopping
> from sklearn.metrics import accuracy_score
> from sklearn.metrics import classification_report
> from sklearn.model_selection import StratifiedKFold
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.linear_model import LogisticRegression
> from sklearn.tree import DecisionTreeClassifier
> from sklearn.naive_bayes import GaussianNB
> from sklearn.neighbors import KNeighborsClassifier
> from catboost import CatBoostClassifier
> import numpy
> import pandas as pd
> import tensorflow as tf
> import matplotlib.pyplot as plt
> import os
> import seaborn as sns
> ```

## 데이터 전처리
> ### 데이터 불러오기
> ```python
>  df = pd.read_csv('heart.csv',index_col=None,header=None)
> ```
> ### LabelEncoder를 사용하여 DataSet에 있는 글자를 숫자로 변형하고 새롭게 저장합니다.
> ```python
> dataset=df.values
> e=LabelEncoder()
> e.fit(dataset[1:,1])
> dataset[1:,1]=e.transform(dataset[1:,1])
> e=LabelEncoder()
> e.fit(dataset[1:,2])
> dataset[1:,2]=e.transform(dataset[1:,2])
> e=LabelEncoder()
> e.fit(dataset[1:,6])
> dataset[1:,6]=e.transform(dataset[1:,6])
> e=LabelEncoder()
> e.fit(dataset[1:,8])
> dataset[1:,8]=e.transform(dataset[1:,8])
> e=LabelEncoder()
> e.fit(dataset[1:,10])
> dataset[1:,10]=e.transform(dataset[1:,10])
> print(df)
> df.to_csv('new_heart.csv')
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191789035-b026ac44-de24-4b5c-a3d4-354ebd3ffa24.png" width="650" height="250">
> 
> 숫자로 대체된 문자
> ```
> Sex : 0 = 여성, 1 = 남성
> CPT :  0 = 무증상(ASY) , 1 = 이례적 협심증(ATA) , 2 = 비협심증 통증(NAP) , 3 = 일반적 협심증(TA)
> RestingECG : 0 = LVH , 1 = Normal , 2 = ST
> ExerciseAngina : 0 = N , 1 = Y
> ST_Slope : 0 = Down , 1 = Flat , 2 = Up
> ```

## 데이터 시각화
> ### 심장병 환자수 시각화
> ```python
> df_new3=pd.read_csv('new_heart2.csv')
> print(df_new3["HeartDisease"].value_counts())
> f=sns.countplot(x='HeartDisease',data=df_new3)
> f.set_xticklabels(['NO Heart Disease','Heart Disease'])
> plt.xlabel("")
> fig,ax=plt.subplots(1, 1, figsize = (7,7))
> df_new3['HeartDisease'].value_counts().plot.pie(explode=[0,0.05],startangle=90, autopct='%0.1f%%',ax=ax,cmap='coolwarm_r')
> plt.title("Heart Disease")
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191790319-f333206c-6db7-446a-a9a1-26ceb28405b5.png" width="250" height="400">

> ### Heatmap 관계도
> ```python
> colormap=plt.cm.gist_heat
> plt.figure(figsize=(12,12))
> sns.heatmap(df_new3.corr(),linewidths=0.1,vmax=0.5,cmap=colormap,linecolor='white',annot =True)
> plt.show()
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191790969-289c40dc-e801-4979-bbd7-787b7854796b.png" width="400" height="400">

> ### 협심증 유발 운동, 가슴통증, 공복 시 혈당, 최대 운동 심전도 기울기, 나이와 심장병의 관계도
> ```python
> f, ax = plt.subplots(2, 2, figsize=(16, 8))
> sns.countplot('ExerciseAngina', hue='HeartDisease', data=df_new3,ax=ax[0,0])
> sns.countplot('ChestPainType', hue='HeartDisease', data=df_new3, ax=ax[0,1])
> sns.countplot('FastingBS', hue='HeartDisease', data=df_new3, ax=ax[1,0])
> sns.countplot('ST_Slope', hue='HeartDisease', data=df_new3, ax=ax[1,1])
> plt.figure(2)
> grid=sns.FacetGrid(df_new3,col='HeartDisease')
> grid.map(plt.hist,'Age',bins=10)
> plt.show()
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191791687-58383c13-44cc-4bab-bf28-824d17389a9d.png" width="425" height="350">

> ### 콜레스테롤, Oldpeak과 심장병의 관계
> ```python
> fig, ax = plt.subplots(1, 2, figsize=(16, 8))
> sns.kdeplot(df_new3[df_new3['HeartDisease']==1]['Cholesterol'], ax=ax[0])
> sns.kdeplot(df_new3[df_new3['HeartDisease']==0]['Cholesterol'], ax=ax[0])
> plt.legend(['HeartDisease', 'NO_HeartDisease'])
> sns.kdeplot(df_new3[df_new3['HeartDisease']==1]['Oldpeak'], ax=ax[1])
> sns.kdeplot(df_new3[df_new3['HeartDisease']==0]['Oldpeak'], ax=ax[1])
> plt.legend(['HeartDisease', 'NO_HeartDisease'])
> plt.show()
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191792306-6d60a8a8-004a-4b85-9f6a-e0379630cad0.png" width="425" height="240">

> ### 성별과 심장병의 관계
> ```python
> print(df_new3["Sex"].value_counts())
> plt.figure(figsize=(8,8))
> sns.countplot('HeartDisease', hue='Sex', data=df_new3)
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191792871-9de6eb3f-67b7-484e-a223-c54287aee4dd.png" width="400" height="400">

> ### 최대심박수, 평상시 혈압과 심장병의 관계
> ```python
> fig, ax = plt.subplots(1, 2, figsize=(16, 8))
> sns.boxplot(data=df_new3, x="HeartDisease", y="MaxHR",ax=ax[0])
> sns.boxplot(data=df_new3, x="HeartDisease", y="RestingBP",ax=ax[1])
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191793200-37141410-e793-4bef-a071-70b7e75455bb.png" width="425" height="240">

> ### 안정 심전도와 심장병의 관계
> ```python
> plt.figure(figsize=(5, 8))
> sns.barplot(data=df_new3 , x="RestingECG", y="HeartDisease")
> ```
> <img src="https://user-images.githubusercontent.com/111839344/191793441-ccc14e60-19c2-4ad0-bf01-63c8d7edfcf2.png" width="250" height="400">

>









