import cv2
import pickle
import sqlite3

# 从OpenCV中加载预训练的人脸级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载用于人脸识别的预训练模型
# 将 'path_to_model' 替换为实际模型文件的路径
model = cv2.face.LBPHFaceRecognizer_create()
model.read('./model.yml')

# 从文件中加载标签字典
with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)

# 创建一个空数组来存储人脸识别结果
recognized_faces = []

def update_balance_by_name(name, new_balance):
    # 连接到数据库
    conn = sqlite3.connect('bank.db')

    # 创建一个游标对象，用于执行SQL语句
    cursor = conn.cursor()

    # 执行更新操作，将给定姓名的账户余额更新为新的余额
    cursor.execute("UPDATE bank SET balance=? WHERE name=?", (new_balance, name))

    # 提交事务
    conn.commit()

    # 输出更新结果
    if cursor.rowcount > 0:
        print(f"{name}帳戶更新後餘額：{new_balance}")

    # 关闭游标和数据库连接
    cursor.close()
    conn.close()


def query_balance_by_name(name):
    # 连接到数据库
    conn = sqlite3.connect('bank.db')

    # 创建一个游标对象，用于执行SQL语句
    cursor = conn.cursor()

    # 执行查询，获取与给定姓名相同的记录
    cursor.execute("SELECT balance FROM bank WHERE name=?", (name,))
    records = cursor.fetchall()

    # 输出查询结果
    if len(records) > 0:
        for record in records:
            print(f"歡迎用戶: {name}, 帳戶餘額: {record[0]}")
            check = input("是否修改帳戶餘額，確定請輸入1：")
            if check == "1":
                money = input("請輸入修改金額：")
                update_balance_by_name(name, money)
    else:
        print("無此用戶.")
        # 询问是否添加新记录
        check= input("是否添加用戶1：")
        if check == "1":
            # 获取新记录的余额
            money = input("請輸入餘額：")
            cursor.execute("INSERT INTO bank (name, balance) VALUES (?, ?)", (name, money))
            conn.commit()
            print(f"已添加帳戶{name},餘額為{money}.")

    # 关闭游标和数据库连接
    cursor.close()
    conn.close()

def recognize_faces(frame):

    flag = True
    # 将图像转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中的人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    name = ""
    
    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        
        # 提取人脸感兴趣区域（ROI）
        face_roi = gray[y:y+h, x:x+w]

        # 将人脸ROI调整为人脸识别模型所需的尺寸
        face_roi = cv2.resize(face_roi, (540, 300))

        # 使用模型预测ROI的标签和置信度
        label, confidence = model.predict(face_roi)

        # 使用标签字典将数字标签转换为人名
        name = label_dict[label]

        # 在图像中显示人名和置信度
        if confidence < 80:
            # 在人脸周围画一个矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f'name: {name}, confidence: {int(confidence)}'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 将人名添加到recognized_faces数组中
            recognized_faces.append(name)

            # 如果检测到的人脸次数达到30次，输出名称
            if recognized_faces.count(name) >= 10:
                flag = False
    return frame,flag,name

cap = cv2.VideoCapture(0) # 默认的摄像头编号是0，如果你有多个摄像头，可以改变这个编号

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if ret:
        # 在图像上进行人脸识别
        frame,flag,name = recognize_faces(frame)
        # 显示处理后的图像
        cv2.imshow('Real-time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        # 按下 'q' 键退出循环
        # 释放摄像头资源
        cap.release()
        # 销毁所有的窗口
        cv2.destroyAllWindows()
        break
    if flag == False:
        # 释放摄像头资源
        cap.release()
        # 销毁所有的窗口
        cv2.destroyAllWindows()
        query_balance_by_name(str(name))
        break
