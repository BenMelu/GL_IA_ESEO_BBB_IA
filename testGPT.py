import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Construire le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")  # binaire : survie ou non
])

# 6. Compiler le modèle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 7. Entraîner le modèle
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8. Évaluer sur les données test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy sur le test set : {accuracy:.2f}")

# 9. Faire une prédiction exemple
sample = np.array([X_test[0]])  # premier passager du test set
prediction = model.predict(sample)
print(sample)
print("Probabilité de survie :", prediction[0][0])
print("Survécu (1) ou pas (0) :", int(prediction[0][0] > 0.5))