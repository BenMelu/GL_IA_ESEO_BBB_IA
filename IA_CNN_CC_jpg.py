import os
from PIL import Image as Img
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter import filedialog
from tensorflow import *
from keras import *
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array

path_directory = os.path.dirname(os.path.abspath(__file__))
imagesDirectory = os.path.join(path_directory, "PetImages")

if os.path.exists(path_directory + "\\IA.keras"):
    model = models.load_model(path_directory + "\\IA.keras")
else:

    count = 0

    for  root, _, files in os.walk(imagesDirectory):
        for f in files:
                count = count + 1
                path = os.path.join(root, f)
                print(count)
                try :
                    with Img.open(path) as img:
                        img = img.convert("RGB")
                        img = img.resize((250, 250), Img.Resampling.LANCZOS)
                        img.save(path)
                except Exception :
                    print("Image corrompue : " + f)
                    os.remove(path)
                print("img : " , count)

    x_train = utils.image_dataset_from_directory(
        imagesDirectory,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=256,
        image_size=(250, 250),
        shuffle=True, seed=42,
        validation_split=0.2,
        subset="training" )

    x_temp = utils.image_dataset_from_directory(
        imagesDirectory,
        labels = "inferred",
        label_mode = "categorical",
        color_mode="grayscale",
        batch_size = 256,
        image_size=(250, 250),
        shuffle=True, seed=42,
        validation_split=0.2,
        subset="validation" )

    val_batches = int(0.5 * data.experimental.cardinality(x_temp).numpy())
    x_val = x_temp.take(val_batches)
    x_test = x_temp.skip(val_batches)

# normalisation des jeux de données
# fait dans la couche de rescaling

# modèle

    nbClass = 2
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(250, 250, 1)),
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(nbClass, activation='softmax')
    ])

    # training du modèle

    early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights = True,
    )
    model.compile(
        optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    historic = model.fit(x_train, epochs=25, validation_data=x_val, callbacks = [early_stop]).history

    model.save(path_directory + "\\IA.keras")

    # test du modèle
    testLoss, testAccuracy = model.evaluate(x_test)

    # affichage des résultats
    acc = historic['accuracy']
    val_acc = historic['val_accuracy']
    loss = historic['loss']
    val_loss = historic['val_loss']

    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    divided = []
    for index, element in enumerate(val_loss):
        divided.append(element / val_acc[index])

    plt.plot(divided, label="val_loss/val_acc")
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

# demander une image à l'utilisateur
while True:
    try :
        tkinterWindow = Tk()
        tkinterWindow.withdraw()
        chatMignonPath = filedialog.askopenfilename(
            title="Sélectionnez une image",
            filetypes=[("Fichiers image", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        print(chatMignonPath)

        # chatMignonPath = os.path.join(path_directory, "imagesTests\\imageAgathe.jpeg")

        with Img.open(chatMignonPath) as chatMignonImg:
            chatMignonImg = chatMignonImg.convert("L")
            chatMignonImg = chatMignonImg.resize((250, 250), Img.Resampling.LANCZOS)
            chatMignon = np.array(chatMignonImg)
            chatMignon = chatMignon
            chatMignon = np.expand_dims(chatMignon, axis=-1)
            chatMignon = np.expand_dims(chatMignon, axis=0)

            prediction = model.predict(x=chatMignon, verbose="auto")

            plt.title(prediction)
            plt.imshow(chatMignonImg)
            plt.show()
    except Exception:
        print("Error try again or juste do a break")
    while input("appuyez sur 'entrée' pour continuer") != "":
        turningLoop = 0