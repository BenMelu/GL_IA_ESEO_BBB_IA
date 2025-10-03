import numpy as np 
import tensorflow as tf
import os.path
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

PATH=os.path.dirname(os.path.realpath(__file__))

X_train=tf.keras.utils.image_dataset_from_directory(directory=PATH+'/dataset/train',labels='inferred',batch_size=64,image_size=(224,224),label_mode='categorical',color_mode="grayscale")
X_test=tf.keras.utils.image_dataset_from_directory(directory=PATH+'/dataset/test',labels='inferred',batch_size=64,image_size=(224,224),label_mode='categorical',color_mode="grayscale")
X_val=tf.keras.utils.image_dataset_from_directory(directory=PATH+'/dataset/val',labels='inferred',batch_size=64,image_size=(224,224),label_mode='categorical',color_mode="grayscale")

if not os.path.isfile(PATH+"/modelShape.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(scale=1./255.,input_shape=(224,224,1)))
    model.add(tf.keras.layers.Conv2D(32,(3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64,(3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128,(3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256,(3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(512,(3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=["accuracy"])
    model.summary()
    history=model.fit(x=X_train,epochs=10,batch_size=64,validation_data=(X_val),callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'], '--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], '--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log',base=10)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    model.save(PATH+"/modelShape.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/modelShape.keras")

result=model.evaluate(X_test,batch_size=128)

pred_1=cv2.imread("test/1C.jpeg",cv2.IMREAD_GRAYSCALE)
pred_1_norm=cv2.resize(pred_1, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
pred_1_norm= 255-pred_1_norm
pred_1_norm[pred_1_norm<150]=0
pred_1_norm = cv2.bitwise_not(pred_1_norm)
test=np.array([pred_1_norm])

def reshapeIM(file,array):
    pred=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    pred_norm=cv2.resize(pred, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    pred_norm= 255-pred_norm
    pred_norm[pred_norm<150]=0
    pred_norm = cv2.bitwise_not(pred_norm)
    arr_pred=np.array([pred_norm])
    array=np.concat((array,arr_pred),axis=0)
    return array

test=reshapeIM("test/2.jpeg",test)
test=reshapeIM("test/3.jpeg",test)
test=reshapeIM("test/4.jpeg",test)
test=reshapeIM("test/5.jpeg",test)
test=reshapeIM("test/6.jpeg",test)
test=reshapeIM("test/7.jpeg",test)
test=reshapeIM("test/8.jpeg",test)
test=reshapeIM("test/9.jpeg",test)
test=reshapeIM("test/10.jpeg",test)
test=reshapeIM("test/11.jpeg",test)
test=reshapeIM("test/12.jpeg",test)
test=reshapeIM("test/13.jpeg",test)
test=reshapeIM("test/14.jpeg",test)
test=reshapeIM("test/15.jpeg",test)
test=reshapeIM("test/16.jpeg",test)
test=reshapeIM("test/17.jpeg",test)
test=reshapeIM("test/18.jpeg",test)
test=reshapeIM("test/19.jpeg",test)
test=reshapeIM("test/20.jpeg",test)
test=reshapeIM("test/21.jpeg",test)
test=reshapeIM("test/22.jpeg",test)
test=reshapeIM("test/23.jpeg",test)
y_pred=model.predict(test)
N=23
class_names = X_train.class_names
y_pred_classes = np.argmax(y_pred, axis=1)
plt.figure(figsize=(10, 10))

for i in range(N):
    plt.subplot(int(np.sqrt(N))+1,int(np.sqrt(N))+1, i + 1)
    plt.imshow(test[i])
    plt.title(f"Prédit: {class_names[y_pred_classes[i]]}") #y_pred_classes[i]
    plt.axis('off')
    
plt.tight_layout()
plt.show()