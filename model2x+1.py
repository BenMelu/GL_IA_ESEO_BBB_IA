import numpy as np
import tensorflow as tf
import os.path

PATH=os.path.dirname(os.path.realpath(__file__))
input1=np.array([1,2,3,4,5,6,7,500,9,10,10000])
output1=np.array([3,5,7,9,11,13,15,1001,19,21,20001])

if not os.path.isfile(PATH+"/model2x+1.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=3,input_shape=[1]))
    model.add(tf.keras.layers.Dense(units=16))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(x=input1,y=output1,epochs=500)

    model.save(PATH+"/model2x+1.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/model2x+1.keras")
    
while True:
    r=int(input('Nombre:'))
    print('prediction : '+str(model.predict(np.array([r]))))
