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

# 載入人臉分類器
classfier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
while(True):
    # 從攝像頭讀取畫面
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在灰度圖像中檢測人臉
    faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:
        # 如果檢測到人臉
        for (x, y, w, h) in faceRects:
            # 提取人臉區域
            face = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            # 在原始畫面上畫出人臉矩形框
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            # 生成一個隨機數字作為圖像檔案名稱
            random_number = random.randint(1000000000, 9999999999)
            # 在檢測到人臉時，將圖像保存到對應的資料夾中
            cv2.imwrite(f'data_dir/{name}/{str(random_number)}.jpg', face)
    
    # 顯示攝像頭畫面
    cv2.imshow('live', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放攝像頭資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
