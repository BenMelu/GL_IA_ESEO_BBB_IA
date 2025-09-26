import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
import os.path

PATH=os.path.dirname(os.path.realpath(__file__))

input1=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
output1=np.array([1,4,9,16,25,36,49,64,81,100,121,144,139,196,225,256,289,324,361,400])

input1_norm=(input1-np.min(input1))/(np.max(input1)-np.min(input1))
output1_norm=(output1-np.min(output1))/(np.max(output1)-np.min(output1))

if not os.path.isfile(PATH+"/modelx2.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1,input_shape=[1]))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(loss='mean_squared_error',optimizer='SGD')

    model.fit(x=input1_norm,y=output1_norm,epochs=500)

    model.save(PATH+"/modelx2.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/modelx2.keras")
    
while True:
    r=int(input('Nombre:'))
    print('prediction : '+str(model.predict(np.array([r]))))
    print(r**2)