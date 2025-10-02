import numpy as np
import tensorflow as tf
import cv2
import os.path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

PATH=os.path.dirname(os.path.realpath(__file__))

(X_train,y_train),(X_test,y_test)= tf.keras.datasets.cifar100.load_data(label_mode="coarse",)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)
print(X_train.shape)
X_train_norm=X_train/255
X_test_norm=X_test/255
X_val_norm=X_val/255

if not os.path.isfile(PATH+"/modelCIFAR100.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(32,32,3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same",strides=1))
    #model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same",strides=1))
    #model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding="same",strides=1))
    #model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    '''model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))'''
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    '''model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))'''
    model.add(tf.keras.layers.Dense(units=20, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=["accuracy"])
    model.summary()
    history=model.fit(x=X_train_norm,y=y_train,epochs=75,batch_size=128,validation_data=(X_val_norm, y_val),callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

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

    model.save(PATH+"/modelCIFAR100.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/modelCIFAR100.keras")

    
result=model.evaluate(X_test_norm,y_test,batch_size=128)
def plot_misclassified_images(model, X_test, y_test, N=1):
    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Classes prédites
    y_true_classes = np.argmax(y_test, axis=1)  # Classes vraies
    
    # Identifier les indices des images mal classifiées
    misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    # Vérifier si le nombre d'images mal classifiées est suffisant
    if len(misclassified_indices) < N:
        print(f"Il n'y a que {len(misclassified_indices)} images mal classifiées.")
        N = len(misclassified_indices)

    # Sélectionner aléatoirement N indices d'images mal classifiées
    random_indices = np.random.choice(misclassified_indices, N, replace=False)
    
    # Afficher les images mal classifiées
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(random_indices):
        plt.subplot(int(np.sqrt(N)), int(np.sqrt(N)) + 1, i + 1)
        plt.imshow(X_test[idx].reshape(32,32,3), cmap='brg') 
        plt.title(f"Prédit: {y_pred_classes[idx]}\nVérité: {y_true_classes[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
plot_misclassified_images(model, X_test_norm, y_test, 25)