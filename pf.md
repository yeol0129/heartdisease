# Heart diease prediction model 

#### code : [click here](https://github.com/yeol0129/xray_ResNet50_Pneumonia/blob/main/pneumonia_resnet50.ipynb)
## Data
> ### example of heart.csv
> Age|Sex|ChestPainType|RestingBP|Cholesterol|FastingBS|RestingECG|MaxHR|ExerciseAngina|Oldpeak|ST_Slope|HeartDisease
> ---|---|---|---|---|---|---|---|---|---|---|---|
> 40|M|ATA|140|289|0|Normal|172|N|0|Up|0
> 49|F|NAP|160|180|0|Normal|156|N|1|Flat|1
> 37|M|ATA|130|283|0|ST|98|N|0|Up|0

## Data preprocessing
### Use the label encoder to change letters into numbers.
>
> <details>
> <summary>open code</summary>
>
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
>
> </details>

After labelencoder : 

<img src="https://user-images.githubusercontent.com/111839344/191789035-b026ac44-de24-4b5c-a3d4-354ebd3ffa24.png" width="650" height="250">

They are letters converted into numbers.
> ```
> Sex : 0 = female, 1 = male
> CPT :  0 = ASY, 1 = ATA, 2 = NAP, 3 = TA
> RestingECG : 0 = LVH, 1 = Normal, 2 = ST
> ExerciseAngina : 0 = N, 1 = Y
> ST_Slope : 0 = Down, 1 = Flat, 2 = Up
> ```

## Data visualization
### Visualization of Heart Disease Number
>
> <details>
> <summary>open code</summary>
>
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
> 
> </details>

<img src="https://user-images.githubusercontent.com/111839344/191790319-f333206c-6db7-446a-a9a1-26ceb28405b5.png" width="250" height="400">

### Heatmap
>
> <details>
> <summary>open code</summary>
>
> ```python
> colormap=plt.cm.gist_heat
> plt.figure(figsize=(12,12))
> sns.heatmap(df_new3.corr(),linewidths=0.1,vmax=0.5,cmap=colormap,linecolor='white',annot =True)
> plt.show()
> ```
>
> </details>
 
<img src="https://user-images.githubusercontent.com/111839344/191790969-289c40dc-e801-4979-bbd7-787b7854796b.png" width="400" height="400">

### Relationship between angina-inducing exercise, chest pain, fasting blood sugar, maximum exercise ECG slope, age and heart disease
>
> <details>
> <summary>open code</summary>
>
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
>
> </details>

<img src="https://user-images.githubusercontent.com/111839344/191791687-58383c13-44cc-4bab-bf28-824d17389a9d.png" width="425" height="350">

### Relationship between cholesterol, oldpeak and heart disease
>
> <details>
> <summary>open code</summary>
>
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
>
> </details>
 
<img src="https://user-images.githubusercontent.com/111839344/191792306-6d60a8a8-004a-4b85-9f6a-e0379630cad0.png" width="425" height="240">

### Relationship between sex and heart disease
>
> <details>
> <summary>open code</summary>
>
> ```python
> print(df_new3["Sex"].value_counts())
> plt.figure(figsize=(8,8))
> sns.countplot('HeartDisease', hue='Sex', data=df_new3)
> ```
>
> </details>
<img src="https://user-images.githubusercontent.com/111839344/191792871-9de6eb3f-67b7-484e-a223-c54287aee4dd.png" width="400" height="400">

### Relationship between maxHR, restingBP and heart disease
>
> <details>
> <summary>open code</summary>
>
> ```python
> fig, ax = plt.subplots(1, 2, figsize=(16, 8))
> sns.boxplot(data=df_new3, x="HeartDisease", y="MaxHR",ax=ax[0])
> sns.boxplot(data=df_new3, x="HeartDisease", y="RestingBP",ax=ax[1])
> ```
>
> </details>
<img src="https://user-images.githubusercontent.com/111839344/191793200-37141410-e793-4bef-a071-70b7e75455bb.png" width="425" height="240">

### Relationship between restingECG and heart disease
> <details>
> <summary>open code</summary>
>
> ```python
> plt.figure(figsize=(5, 8))
> sns.barplot(data=df_new3 , x="RestingECG", y="HeartDisease")
> ```
>
> </details>
> <img src="https://user-images.githubusercontent.com/111839344/191793441-ccc14e60-19c2-4ad0-bf01-63c8d7edfcf2.png" width="250" height="400">

## Modeling
### Separate data into training(80%) and validation(20%).학습셋과 테스트 셋의 구분. 테스트셋 20%, 학습셋 80%
> <details>
> <summary>open code</summary>
>
> ```python
> X=dataset[2:,1:12]
> Y=dataset[2:,12]
> X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=seed)
> print(X_train.shape)
> print(X_test.shape)
> ```
>
> </details>
> output : 
> ```
> (734, 11)
> (184, 11)
> ```

### Keras Sequential Model
> <details>
> <summary>open code</summary>
>
 > ```python
 > model=Sequential()
 > model.add(Dense(64, input_dim=11,activation='relu'))    
 > model.add(Dense(32, activation='relu'))                
 > model.add(Dense(16, activation='relu'))               
 > model.add(Dense(1,activation='sigmoid'))
 > ...
 > hist=model.fit(X,Y,validation_split=0.20,epochs=350,batch_size=500,callbacks=[early_stopping_callback,checkpointer])
 > ```
>
> </details>
> 
 > >output : 
 > >```
 > >Epoch 00321: val_loss did not improve from 0.40818
> >Epoch 322/350
> >734/734 [==============================] - 0s 7us/step - loss: 0.3141 - acc: 0.8774 - val_loss: 0.4153 - val_acc: 0.8098
>> 
>> Epoch 00322: val_loss did not improve from 0.40818
>> Epoch 323/350
>> 734/734 [==============================] - 0s 8us/step - loss: 0.3148 - acc: 0.8692 - val_loss: 0.4127 - val_acc: 0.8152
>> 
>> Epoch 00323: val_loss did not improve from 0.40818
>> 918/918 [==============================] - 0s 15us/step          
>> ```
>> ### 케라스 모델 정확도와 손실 시각화
>> ```python
>> print("\n Accuracy:%.4f"%(model.evaluate(X,Y)[1]))  
>> y_vloss=hist.history['val_loss']
>> y_acc=hist.history['acc']       
>> x_len=numpy.arange(len(y_acc))
>> plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
>> plt.plot(x_len,y_acc,"o",c="blue",markersize=3)
>> plt.ylim([0,1])
>> plt.show
>> ```
>> <img src="https://user-images.githubusercontent.com/111839344/191796153-d0819fb1-a2b0-4f2f-8267-2f11a0927e47.png" width="400" height="400">
>>
> ### K겹 교차 검증 모델 (5겹)
>> ```python
>> n_fold=5
>> skf=StratifiedKFold(n_splits=n_fold,shuffle=True, random_state=seed)
>> ...
>> for train,test in skf.split(X,Y):
>>    model_k=Sequential()
>>    model_k.add(Dense(64, input_dim=11,activation='relu'))    
>>    model_k.add(Dense(32, activation='relu'))                 
>>    model_k.add(Dense(16, activation='relu'))                 
>>    model_k.add(Dense(1,activation='sigmoid'))
>> ...
>> model_k.fit(X[train],Y[train],epochs=100,batch_size=5)
>> ...
>> ```
>> output :
>> ```
>> Epoch 100/100
>> 735/735 [==============================] - 0s 160us/step - loss: 0.3351 - acc: 0.8463
>> 183/183 [==============================] - 0s 1ms/step
>> 
>> 5 Fold Accuracy: ['0.8641', '0.8098', '0.5543', '0.8579', '0.8579']
>> ```
> ### 로지스틱 회귀 모델
> > ```python
> > accuracies={}
>>lr = LogisticRegression()
>>lr.fit(X_train,Y_train)
>>```
>> ### 로지스틱 회귀 모델 성능평가
>>```python
>>lr_pred = lr.predict(X_test)
>>acc = lr.score(X_test,Y_test)
>>accuracies['Logistic Regression'] = acc
>>print('Classification report\n',classification_report(Y_test, lr_pred))
>>print("Accuracy of Logistic Regression: {:.2f}".format(acc))
>>```
>>output : 
>>```
>>Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.83      0.77      0.80        77
>>          1       0.84      0.89      0.86       107
>>
>>avg / total       0.84      0.84      0.84       184
>>
>>Accuracy of Logistic Regression: 0.84
>>```
>> ### 가우시안 나이브 베이즈 모델
>> ```python
>> nb = GaussianNB()
>> nb.fit(X_train, Y_train)
>>```
>> ### 가우시안 나이브 베이즈 모델 성능 평가 output : 
>> ```
>>Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.83      0.81      0.82        77
>>          1       0.86      0.88      0.87       107
>>
>>avg / total       0.85      0.85      0.85       184
>>
>>Accuracy of Naive Bayes: 0.85
>>```
> ### 랜덤포레스트 분류 모델
>> ```python
>> rfc=RandomForestClassifier()
>> rfc.fit(X_train, Y_train)
>> ```
>> ### 랜덤포레스트 분류 모델 성능평가 output : 
>> ```
>> Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.85      0.81      0.83        77
>>          1       0.86      0.90      0.88       107
>>
>>avg / total       0.86      0.86      0.86       184
>>
>>Accuracy of Random Forest Classifier: 0.86
>> ```
> ### 결정트리 분류 모델
>> ```python
>> dtc = DecisionTreeClassifier()
>> dtc.fit(X_train, Y_train)
>> ```
>> ### 결정트리 분류 모델 성능평가 output : 
>>```
>>Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.67      0.81      0.73        77
>>          1       0.84      0.72      0.77       107
>>
>>avg / total       0.77      0.76      0.76       184
>>
>>Accuracy of Decision Tree Classifier: 0.76
>>```
> ### K-NN 분류 모델
> >```python
> >knn = KNeighborsClassifier(n_neighbors=10)
>>knn.fit(X_train , Y_train)
>>```
>> ### K-NN 분류 모델 성능평가 output :
>> ```
>> Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.62      0.71      0.67        77
>>          1       0.77      0.69      0.73       107
>>
>>avg / total       0.71      0.70      0.70       184
>>
>>Accuracy of KNN: 0.70
>> ```
> ### CatBoost 분류 모델
>> ```python
>> cb = CatBoostClassifier(iterations=100)
>> cb.fit(X_train, Y_train)
>> ```
>> ### CatBoost 분류 모델 성능 평가 output : 
>> ```
>> Classification report
>>              precision    recall  f1-score   support
>>
>>          0       0.87      0.79      0.83        77
>>          1       0.86      0.92      0.89       107
>>
>>avg / total       0.86      0.86      0.86       184
>>
>>Accuracy of CatBoostClassifier: 0.86
>> ```
 ### K 교차 검증 모델을 이용해 가상환자 데이터의 심장질환 확률 예측
 ```python
 #협심증유발운동 유, 50대 이상, 남성, 최대심박수 140이하, 콜레스테롤 낮음
#가슴통증 없음, oldpeak 0이상, 공복시혈당 True일경우 심장질환확률 높을것으로 예상 
# 가상의 환자 데이터 입력
#Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
patient = numpy.array([[55,1,0,160,120,1,0,130,1,1,0]])
# k교차검증모델로 예측
pred = model_k.predict(patient)
# 예측결과 출력
print(pred*100)
```
output : 
```
[[94.81547]]
```
