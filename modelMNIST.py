import numpy as np
import tensorflow as tf
import cv2
import os.path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

PATH=os.path.dirname(os.path.realpath(__file__))

(X_train,y_train),(X_test,y_test)= tf.keras.datasets.mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)
print(X_train.shape)
X_train_norm=X_train/255
X_test_norm=X_test/255
X_val_norm=X_val/255

if not os.path.isfile(PATH+"/modelMNIST.keras"):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,(3,3),input_shape=(28,28,1), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same",strides=1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

    history=model.fit(x=X_train_norm,y=y_train,epochs=50,batch_size=256,validation_data=(X_val_norm, y_val))

    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'], '--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1)
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

    model.save(PATH+"/modelMNIST.keras")
    
else:
    model=tf.keras.models.load_model(PATH+"/modelMNIST.keras")

    
result=model.evaluate(X_test_norm,y_test,batch_size=128)
'''def plot_misclassified_images(model, X_test, y_test, N=1):
    """
    Affiche N images mal classifiées par le modèle donné.

    Parameters:
    - model: Le modèle pré-entraîné.
    - X_test: Les images de test.
    - y_test: Les étiquettes de test (one-hot encodées).
    """
    
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
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')  # Assure que les images sont en 28x28
        plt.title(f"Prédit: {y_pred_classes[idx]}\nVérité: {y_true_classes[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
plot_misclassified_images(model, X_test_norm, y_test, 25)'''

pred_9=cv2.imread("photos_MNIST/9G.jpeg",cv2.IMREAD_GRAYSCALE)
pred_9_norm=cv2.resize(pred_9, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
pred_9_norm = cv2.bitwise_not(pred_9_norm)

test=np.array([pred_9_norm])
def reshapeIM(file,array):
    pred=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    pred_norm=cv2.resize(pred, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    pred_norm = cv2.bitwise_not(pred_norm)
    arr_pred=np.array([pred_norm])
    array=np.concat((array,arr_pred),axis=0)
    return array

test=reshapeIM("photos_MNIST/8G.jpeg",test)
test=reshapeIM("photos_MNIST/7G.jpeg",test)
test=reshapeIM("photos_MNIST/6G.jpeg",test)
test=reshapeIM("photos_MNIST/5G.jpeg",test)
test=reshapeIM("photos_MNIST/4G.jpeg",test)
test=reshapeIM("photos_MNIST/3G.jpeg",test)
test=reshapeIM("photos_MNIST/2G.jpeg",test)
test=reshapeIM("photos_MNIST/1G.jpeg",test)
y_pred=model.predict(test)
N=3
y_pred_classes = np.argmax(y_pred, axis=1)
plt.figure(figsize=(12, 12))
for i in range(0,9):
        plt.subplot(3,3, i + 1)
        plt.imshow(test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Prédit: {y_pred_classes[i]}")
        plt.axis('off')
    
plt.tight_layout()
plt.show()
