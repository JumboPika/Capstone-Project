import cv2 as cv
import numpy as np
import socket
import pygame

# 初始化Socket接收
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('172.17.38.114', 5005))

# 初始化 Pygame 來播放聲音
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

# 顯示警示視窗的函數，依照不同的狀態設定顏色
def show_alert_window(color, text):
    alert_window = np.zeros((200, 400, 3), dtype=np.uint8)
    alert_window[:] = color  # 設定視窗顏色
    cv.putText(alert_window, text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow("Alert", alert_window)

while True:
    data, _ = sock.recvfrom(1024)
    message = data.decode()

    if message == 'sitting_up':
        # 顯示橘色視窗，不撥放警報音
        show_alert_window((0, 140, 255), '坐起')
        print("偵測到坐起")

    elif message == 'turning_waist':
        # 顯示紅色視窗，並撥放警報音
        show_alert_window((0, 0, 255), '轉腰 - 警報!')
        pygame.mixer.Sound.play(alarm_sound)
        print("偵測到轉腰")

    elif message == "No person detected":
        # 顯示綠色視窗表示未偵測到人
        show_alert_window((0, 255, 0), '未偵測到人')
        print("未偵測到人")

    if cv.waitKey(1) == 27:  # 按下ESC鍵退出
        break

cv.destroyAllWindows()
pygame.quit()
