import numpy as np
import tensorflow as tf
import cv2
from tkinter import Tk, filedialog
import os.path
from matplotlib import pyplot as plt

PATH=os.path.dirname(os.path.realpath(__file__))

model=tf.keras.models.load_model(PATH+"/modelMNIST.keras")
flag=True

while flag:
    root=Tk()
    root.withdraw()
    filepath=filedialog.askopenfilename(title="Choisir une image",filetypes=[("Images","*.jpg;*.png;*.jpeg;*.bmp")])
    if filepath:
        pred=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        pred_norm=cv2.resize(pred, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        pred_norm = cv2.bitwise_not(pred_norm)
        test=np.array([pred_norm])
        y_pred=model.predict(test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        plt.figure(figsize=(6,6))
        plt.imshow(test[0])
        plt.title(f"Prédit: {y_pred_classes[0]}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune image sélectionnée.")
        flag=False