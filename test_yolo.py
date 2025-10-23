
import ultralytics as u
import os.path
import cv2

PATH=os.path.dirname(os.path.realpath(__file__))
PATH_MODEL="C:/Users/Lilian/runs/detect/train10/weights"

if not os.path.isfile(PATH_MODEL+"/best.pt"):
    model=u.YOLO("yolo11n.pt")
    result=model.train(data=PATH+"/data/data.yaml",epochs=30,patience=3)
else:
    model=u.YOLO(PATH_MODEL+"/best.pt")

#cam=cv2.VideoCapture(0)
img=cv2.imread(PATH+"/test/quiz_reviser_panneaux_signalisation_route_france.jpg")
#while (True):
    #ret,frame=cam.read()
    #if not ret:
    #    break
detection=model(img)
for bbox in detection[0].boxes:
    x1,y1,x2,y2=bbox.xyxy[0]
    class_name=detection[0].names[int(bbox.cls[0])]
    conf = float(bbox.conf[0])
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
    cv2.putText(img,f"{class_name}: {conf:.2f}",(int(x1), max(int(y1) - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,255), 1)
while True:
    cv2.imshow("test",img)
    if cv2.waitKey(1)==ord('q'):
        break

#cam.release()
#cv2.destroyAllWindows