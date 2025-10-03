import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os.path

PATH=os.path.dirname(os.path.realpath(__file__))

data=pd.read_csv('./data.csv',index_col=0)
data.drop(columns=['Ticket', 'Cabin'], inplace=True)
data['Survived']=data['Survived'].astype('bool')
data['Pclass']=data['Pclass'].astype('category')
data.drop(data[data['Age']<0].index, inplace=True)
data.drop(data[data['Age']==138].index, inplace=True)
data['Fare']=data['Fare'].round(2);data['LWeight']=data['LWeight'].round(1)
data.drop(columns = ['Name'], inplace=True)
median_sex=data.groupby('Sex')['Age'].median()
data['Age']=data.apply(lambda x: median_sex[x['Sex']] if pd.isna(x['Age']) else x['Age'],axis=1)
mostCommonEmb = data['Embarked'].mode()[0]
data['Embarked'] = data['Embarked'].fillna(mostCommonEmb)
data['Sex']=data.apply(lambda x: 0 if x['Sex']=='male' else 1,axis=1)
data['Embarked']=data.apply(lambda x: 0 if x['Embarked']=='S' else (1 if x['Embarked']=='C' else 2),axis=1)



X=(data.drop(columns=['Survived'])).astype('float64')
Y=data['Survived']

X_pred_R = pd.DataFrame({"Pclass":[1.0],"Sex":[1.0],"Age":[17.0],"SibSp":[1.0],"Parch":[1.0],"Fare":[56.0],"Embarked":[0.0],"LWeight":[2.5]})
X_pred_J = pd.DataFrame({"Pclass":[3.0],"Sex":[0.0],"Age":[20.0],"SibSp":[0.0],"Parch":[0.0],"Fare":[7.85],"Embarked":[0.0],"LWeight":[1.1]})

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
X_pred=X_train[:1]
scaler=StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_val_norm=scaler.transform(X_val)
X_pred_norm=scaler.transform(X_pred)
X_pred_norm_R=scaler.transform(X_pred_R)
X_pred_norm_J=scaler.transform(X_pred_J)

if not os.path.isfile(PATH+"/modelTita.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=128,input_dim=X_train.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='SGD',metrics=["accuracy"])

    model.fit(x=X_train_norm,y=y_train,epochs=500,validation_data=(X_val_norm, y_val),callbacks=[tf.keras.callbacks.EarlyStopping(patience=25)])

    model.save(PATH+"/modelTita.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/modelTita.keras")
    
result=model.evaluate(X_test_norm,y_test,batch_size=32)
print(result)
print(X_pred)
pred=model.predict(X_pred_norm)
print(pred)
print(model.predict(X_pred_norm_R))
print(model.predict(X_pred_norm_J))