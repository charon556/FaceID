import cv2
import os
import numpy as np

# 函數：從指定路徑加載訓練數據集
def load_training_data(data_dir):
    images = []
    labels = []
    label_dict = {}

    # 獲取數據集中的子目錄（每個子目錄代表一個人）
    subdirs = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

    for i, subdir in enumerate(subdirs):
        label_dict[i] = subdir
        subdir_path = os.path.join(data_dir, subdir)

        # 加載每個子目錄中的圖像文件
        for filename in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, filename)

             # 讀取圖像並將其轉換為灰度
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 將圖像和標籤添加到訓練集中
            images.append(image)
            labels.append(i)

    return images, labels, label_dict

import pickle

def train_face_recognizer(data_dir, model_path):
    # 加載訓練數據集
    images, labels, label_dict = load_training_data(data_dir)

    # 創建LBPH人臉辨識器
    model = cv2.face.LBPHFaceRecognizer_create()

    # 訓練模型
    model.train(images, np.array(labels))

    # 將模型儲存到文件中
    model.save(model_path)

    # 將標籤字典保存到文件中
    with open('label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)

    print("模型訓練完成！")


# 設置數據集目錄和模型文件路徑
data_dir = './data_dir'
model_path = './model.yml'

# 訓練人臉辨識模型
train_face_recognizer(data_dir, model_path)
