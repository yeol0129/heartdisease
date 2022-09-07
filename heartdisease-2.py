#!/usr/bin/env python
# coding: utf-8

# In[71]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy
import pandas as pd
import tensorflow as tf
# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(3)

#csv파일 불러온다
df = pd.read_csv('heart.csv',header=None)

'''

print(df.info())

print(df.head())

'''

#sex,chestpain,restingec,exercise,heartdiseas의 문자를 숫자로 변환
dataset=df.values
e=LabelEncoder()
e.fit(dataset[1:,1])
dataset[1:,1]=e.transform(dataset[1:,1])
e=LabelEncoder()
e.fit(dataset[1:,2])
dataset[1:,2]=e.transform(dataset[1:,2])
e=LabelEncoder()
e.fit(dataset[1:,6])
dataset[1:,6]=e.transform(dataset[1:,6])
e=LabelEncoder()
e.fit(dataset[1:,8])
dataset[1:,8]=e.transform(dataset[1:,8])
e=LabelEncoder()
e.fit(dataset[1:,10])
dataset[1:,10]=e.transform(dataset[1:,10])

print(df)
#숫자로 변환한것을new_heart.csv로 저장
df.to_csv('new_heart.csv')


# In[72]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(3)
#new_heart.csv를 불러온다
df=pd.read_csv('new_heart.csv',header=None)
dataset=df.values
#age와sex를 제외한 나머지 저장
X=dataset[2:,1:12]
Y=dataset[2:,12]
#학습셋과 테스트 셋의 구분. 테스트셋 30%, 학습셋 70%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,random_state=seed)

model=Sequential()
model.add(Dense(64, input_dim=11,activation='relu'))    #12개의 값 받아 24개 노드로 보낸다.활성화함수=relu 은닉층
model.add(Dense(32, activation='relu'))                 #은닉층
model.add(Dense(16, activation='relu'))                 #은닉층
model.add(Dense(1,activation='sigmoid'))                #노드1개, 활성화함수=sigmoid출력층

model.compile(loss='binary_crossentropy',                #컴파일, 오차mean_squared_error,최적화 adam
             optimizer='adam',
             metrics=['accuracy'])


#모델 저장 폴더 설정
MODEL_DIR='./model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
#모델 저장 조건 설정
modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
#앞서 저장한 모델보다 나아졌을 때만 저장
checkpointer=ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)


#학습자동중단설정
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=100)

hist=model.fit(X,Y,validation_split=0.20,epochs=3500,batch_size=500,callbacks=[early_stopping_callback,checkpointer])
#빨간색 손실,파란색 예측 표로 나타내기
y_vloss=hist.history['val_loss'] #테스트셋으로 실험결과 오차값 저장

y_acc=hist.history['acc']        #학습셋으로 측정한 정확도값 저장

x_len=numpy.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)
plt.ylim([0,1])
plt.show


# In[76]:


# 가상의 환자 데이터 입력
patient = numpy.array([[70,0,0,160,280,1,2,156,0,1.5,1]])

# 모델로 예측하기
pred = model.predict(patient)

# 예측결과 출력하기
print(pred*100)


# In[ ]:





# In[ ]:




