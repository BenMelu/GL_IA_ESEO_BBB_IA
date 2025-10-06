
import ultralytics as u
import os.path
import cv2

PATH=os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    model=u.YOLO("yolo11n.pt")

    result=model.train(data=PATH+"/data/data.yaml",epochs=30)


cam=cv2.VideoCapture(0)

while (True):
    ret,frame=cam.read()
    if not ret:
        break
    detection=model(frame)
    for bbox in detection[0].boxes:
        x1,y1,x2,y2=bbox.xyxy[0]
        class_name=detection[0].names[int(bbox.cls[0])]
        conf = float(bbox.conf[0])
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
        cv2.putText(frame,f"{class_name}: {conf:.2f}",(int(x1), max(int(y1) - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,255), 1)
    cv2.imshow("test",frame)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows