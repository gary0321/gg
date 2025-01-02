# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 06:41:04 2024

@author: gary0
"""
# pyQT
import sys
import cv2
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from demo_ui import Ui_MainWindow
from PyQt5.QtCore import QTimer

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 操作
# 設定工作目錄為程式實際所在資料夾
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hand_detect import NumberofFingers
from to_model import test_image

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 初始化介面
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化分數
        self.score = 0
        self.countdown = 10

        # 初始化攝影機
        self.cap = cv2.VideoCapture(0)

        #初始化畫布
        self.drawing = False #紀錄變數畫畫開始變數
        self.startdrawing = False
        self.positions_x=[] #紀錄食指位置x[8]變數
        self.positions_y=[] #紀錄食指位置y[8]變數
        self.pred_class = None
        self.count = 0
        self.paintWindow = self.load_black_canvas()
        self.pred_class = 0
        
        # 用於更新攝影機畫面的計時器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # 每30毫秒更新一次

        self.number_timer = QtCore.QTimer()
        self.number_timer.timeout.connect(self.simulate_number_input)
        self.number_timer.start(10000)  # 每10秒隨機產生一個數字

        self.minus_timer = QtCore.QTimer()
        self.minus_timer.timeout.connect(self.minus_countdown)
        self.minus_timer.start(1000)  # 每1秒倒數一次

        # 初始化第一張圖
        self.display_image_in_view(fr'picture1.png')  # 替換為您的圖片路徑
        self.random_number = 1

    # 圖片變換倒數計時
    def minus_countdown(self):
        self.countdown -= 1
        self.update_labels(self.score, self.countdown)

    # 更新攝影機
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 水平翻轉畫面（鏡像效果）
            frame = cv2.flip(frame, 1)

            """更新手部辨識與繪圖視窗"""
            image, handimage, finger, indexfinger = NumberofFingers(self.cap)

            # 判斷手勢並啟動繪圖模式
            if finger == 1:  # 食指伸出
                if self.count < 10 and not self.startdrawing:
                    self.count += 1
                else:
                    self.count = 0
                    self.drawing = True
                    self.startdrawing = True
            elif finger == 5 and self.startdrawing:  # 五指伸出，結束繪圖
                self.save_and_clear_canvas()
            else:
                self.drawing = False

            # 繪圖邏輯
            if self.drawing and finger == 1:
                self.positions_x.append(indexfinger[0])
                self.positions_y.append(indexfinger[1])

            # 在畫布上畫出軌跡
            self.draw_on_canvas(frame)

            # 顯示結果
            cv2.imshow("paint", self.paintWindow)

            # 將畫面轉換為RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # 設定攝影機畫面為背景
            self.set_background(pixmap)

            # 按 'q' 結束
            if cv2.waitKey(5) & 0xFF == ord('q'):
                self.close()

    #用於動態更新 QLabel 的文字
    def update_labels(self, score, countdown):
        self.ui.label.setText(f"分數: {score}")
        self.ui.label_2.setText(f"圖片變換倒數: {countdown} 秒")

    # 顯示手指路徑的畫布
    def load_black_canvas(self):
        """載入黑色畫布，若載入失敗則建立新畫布"""
        path = 'black.jpg'
        paintWindow = cv2.imread(path)
        if paintWindow is None:
            print("找不到 black.jpg，或載入失敗，建立新畫布")
            paintWindow = np.zeros((600, 600, 3), dtype=np.uint8)  # 建立黑色畫布
        return paintWindow

    # 圖片辨識與重製畫布
    def save_and_clear_canvas(self):
        """儲存畫布內容並重置畫布"""
        xmin = max(0, int(min(self.positions_x)) - 10)
        xmax = min(600, int(max(self.positions_x)) + 10)
        ymin = max(0, int(min(self.positions_y)) - 10)
        ymax = min(600, int(max(self.positions_y)) + 10)

        cropped = self.paintWindow[ymin:ymax, xmin:xmax]
        cv2.imwrite("newestpaint.png", cropped)
        predicted_class = test_image("newestpaint.png")
        categories = ['circle', 'envelope', 'triangle', 'star', 'square']
        predicted_label = categories[predicted_class]
        print(f"預測結果為:{predicted_label}")
        if (predicted_class+1) == (self.random_number):
            self.ui.label.setStyleSheet("background-color: #d2fcd9;")
            self.ui.label_3.setStyleSheet("color: green;font-weight: bold;")
            self.ui.label_3.setText(f"正確")
            self.simulate_number_input(1)
        else:
            self.ui.label.setStyleSheet("background-color: #fcd2f5;")
            self.ui.label_3.setStyleSheet("color: red;font-weight: bold;")  # 將文字顏色改為紅色
            self.ui.label_3.setText(f"錯誤")
            self.simulate_number_input(-1)
        self.number_timer.start(10000)
        
        # 清空畫布
        self.drawing = False
        self.startdrawing = False
        self.positions_x = []
        self.positions_y = []
        self.paintWindow = self.load_black_canvas()

    # 在畫面上繪圖
    def draw_on_canvas(self,frame):
        """在畫布和即時畫面上繪圖"""
        if len(self.positions_x) > 0:
            for i in range(len(self.positions_x)):
                cv2.rectangle(frame, (int(self.positions_x[i]), int(self.positions_y[i])),
                              (int(self.positions_x[i] + 3), int(self.positions_y[i] + 3)), (0, 0, 0), -1)
                cv2.rectangle(self.paintWindow, (int(self.positions_x[i]), int(self.positions_y[i])),
                              (int(self.positions_x[i] + 3), int(self.positions_y[i] + 3)), (255, 255, 255), -1)
            # 繪製線條
            if len(self.positions_x) > 1:
                cv2.line(self.paintWindow,
                         (int(self.positions_x[-1]), int(self.positions_y[-1])),
                         (int(self.positions_x[-2]), int(self.positions_y[-2])),
                         (255, 255, 255), 5)
                cv2.line(frame,
                         (int(self.positions_x[-1]), int(self.positions_y[-1])),
                         (int(self.positions_x[-2]), int(self.positions_y[-2])),
                         (255, 0, 0), 3)

    # 設定鏡頭為背景
    def set_background(self, pixmap):
        # 將pixmap縮放至主視窗大小
        scaled_pixmap = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatioByExpanding)
        palette = self.palette()
        palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(scaled_pixmap))
        self.setPalette(palette)

    # 變換左上角圖片
    def display_image_in_view(self, image_path):
        # 載入圖片
        pixmap = QPixmap(image_path)

        # 縮小圖片並保持比例，縮小到不超過 QGraphicsView 的大小，額外縮小 10%
        target_size = self.ui.graphicsView.size() * 0.9  # 讓圖片縮小一點
        scaled_pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio)

        # 建立場景並將圖片添加到QGraphicsView中
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(scaled_pixmap)
        self.ui.graphicsView.setScene(scene)

    # 隨機變換圖片
    def simulate_number_input(self,plus_number=0):
        self.random_number = random.randint(1, 5)
        self.display_image_in_view(fr'picture{self.random_number}.png')

        # 清空畫布
        self.drawing = False
        self.startdrawing = False
        self.positions_x = []
        self.positions_y = []
        self.paintWindow = self.load_black_canvas()

        # 假設分數隨機變化，倒數時間固定為 10 秒
        self.score += plus_number
        if self.score<=0:
            self.score = 0
        self.countdown = 10
        self.update_labels(self.score, self.countdown)

    # 關閉程式
    def closeEvent(self, event):
        # 關閉應用程式時釋放攝影機
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



