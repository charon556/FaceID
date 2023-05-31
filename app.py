import tkinter as tk
import os
import subprocess

def run_inputFace():
    subprocess.call(["python", "inputFace.py"])

def run_model():
    subprocess.call(["python", "model.py"])

def run_faceID():
    subprocess.call(["python", "faceID.py"])

# 建立視窗
window = tk.Tk()
window.title("程式功能")

# 建立按鈕並設定其命令
inputFace_button = tk.Button(window, text="輸入人臉", command=run_inputFace)
model_button = tk.Button(window, text="訓練模型", command=run_model)
faceID_button = tk.Button(window, text="人臉辨識", command=run_faceID)

# 放置按鈕到視窗中
inputFace_button.pack()
model_button.pack()
faceID_button.pack()

# 開始運行視窗應用程式
window.mainloop()
