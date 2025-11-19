import ultralytics as u
import os.path
import cv2
from tkinter import Tk,filedialog

PATH=os.path.dirname(os.path.realpath(__file__))
PATH_MODEL=PATH+"/weights"

if not os.path.isfile(PATH_MODEL+"/best.pt"):
    model=u.YOLO("yolo11n.pt")
    result=model.train(data=PATH+"/data/data.yaml",epochs=50,patience=5,imgsz=800)
else:
    model=u.YOLO(PATH_MODEL+"/best.pt")

root=Tk()
root.withdraw()
while True:
    filepath=filedialog.askopenfilename(title="Choisir une image",filetypes=[("Images","*.jpg;*.png;*.jpeg;*.bmp")])
    img=cv2.imread(filepath)
    img = cv2.resize(img, (800, 600))
    detection=model(img)
    for bbox in detection[0].boxes:
        x1,y1,x2,y2=bbox.xyxy[0]
        class_name=detection[0].names[int(bbox.cls[0])]
        conf = float(bbox.conf[0])
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
        cv2.putText(img,f"{class_name}: {conf:.2f}",(int(x1), max(int(y1) - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,255), 1)
    cv2.imshow("test",img,)
    if cv2.waitKey(1)==ord('q'):
        break