# 머신러닝 심장질환 예측
=============
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
> <img src="https://user-images.githubusercontent.com/111839344/191790319-f333206c-6db7-446a-a9a1-26ceb28405b5.png" width="170" height="300">


