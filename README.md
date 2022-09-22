# 머신러닝 심장질환 예측
=============
#### 자세한 코드는 [여기있습니다](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb)
## Data
> ## heart.csv의 데이터 예
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

