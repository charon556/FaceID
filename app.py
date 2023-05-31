import tkinter as tk
import os
import subprocess

# 執行 inputFace.py 腳本的函數
def run_inputFace():
    subprocess.call(["python", "inputFace.py"])

# 執行 model.py 腳本的函數
def run_model():
    subprocess.call(["python", "model.py"])

# 執行 faceID.py 腳本的函數
def run_faceID():
    subprocess.call(["python", "faceID.py"])

# 建立一個視窗
window = tk.Tk()
window.title("程式功能")  # 設定視窗標題

# 建立按鈕並設定其命令
inputFace_button = tk.Button(window, text="輸入人臉", command=run_inputFace)  # 按鈕，執行 inputFace.py
model_button = tk.Button(window, text="訓練模型", command=run_model)  # 按鈕，執行 model.py
faceID_button = tk.Button(window, text="人臉辨識", command=run_faceID)  # 按鈕，執行 faceID.py

# 放置按鈕到視窗中
inputFace_button.pack()
model_button.pack()
faceID_button.pack()

# 開始運行視窗應用程式
window.mainloop()
