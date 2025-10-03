import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os.path

PATH=os.path.dirname(os.path.realpath(__file__))
model=tf.keras.models.load_model(PATH+"/modelTita.keras")

flag=True

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

while flag:
    classe=int(input("Dans quelle classe voyager vous : "))
    sex=str(input("Veuillez donner votre sexe (m ou f): "))
    sex=(lambda x: 0 if x=='m' else 1)(sex)
    age=int(input("Veuillez donner votre age : "))
    siblings=int(input("Avec combien de frères, soeurs et/ou partenaire voyager vous : "))
    parents=int(input("Avec combien de parents autres voyager vous : "))
    prix=float(input("Quel est le prix du ticket de votre voyage : "))
    embarked=str(input("Dans quel ports embarqué vous (S, C ou Q): "))
    embarked=(lambda x: 0 if x=='S' else (1 if x=='C' else 2))(embarked)
    poidBagage=float(input("Quel est le poid de votre bagage : "))

    X_pred = pd.DataFrame({"Pclass":[classe],"Sex":[sex],"Age":[age],"SibSp":[siblings],"Parch":[parents],"Fare":[prix],"Embarked":[embarked],"LWeight":[poidBagage]})
    X_pred.astype('float64',False)

    scaler=StandardScaler()
    scaler.fit_transform(X_train)
    X_pred_norm=scaler.transform(X_pred)

    pred=model.predict(X_pred_norm)
    print(f"Vous avez un taux de survie de {pred[0][0]*100:.2f}%")
    

