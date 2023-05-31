import cv2
import random

cap = cv2.VideoCapture(0)
while(not cap.isOpened()):
    cap = cv2.VideoCapture(0)
    
import os

# 請求用戶輸入名字
name = input("請輸入您的名字：")

# 創建名字對應的資料夾
folder_path = os.path.join('data_dir', name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

classfier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))

    if len(faceRects) > 0:      
        for (x, y, w, h) in faceRects:
            face = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
            random_number = random.randint(1000000000, 9999999999)
            # 在偵測到人臉時，保存圖像到相應的資料夾中
            cv2.imwrite(f'data_dir/{name}/{str(random_number)}.jpg', face)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
    cv2.imshow('live', frame)


cap.release()
cv2.destroyAllWindows()