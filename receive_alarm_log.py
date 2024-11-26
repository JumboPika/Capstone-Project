import cv2 as cv
import numpy as np
import socket
import pygame
from datetime import datetime, timedelta

# 設定多播參數
MULTICAST_GROUP = '224.0.0.1'
MULTICAST_PORT = 4003

pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

# 日誌檔案
LOG_FILE = "detection_log.txt"

# 初始化上一個動作和最後偵測時間
last_logged_action = None
last_detection_time = datetime.now()

def log_detection(action):
    """記錄偵測動作到日誌檔案，避免重複記錄"""
    global last_logged_action
    if action != last_logged_action:  # 檢查是否為重複動作
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"{current_time}\t{action}\n")
        last_logged_action = action  # 更新最後記錄的動作

def show_alert_window(color, text):
    """顯示警報視窗"""
    alert_window = np.zeros((200, 400, 3), dtype=np.uint8)
    alert_window[:] = color
    cv.putText(alert_window, text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow("Alert", alert_window)

# 設定socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MULTICAST_PORT))
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton('0.0.0.0'))

while True:
    try:
        sock.settimeout(1)  # 設置接收數據超時
        data, _ = sock.recvfrom(1024)
        message = data.decode()

        # 更新最後偵測時間
        last_detection_time = datetime.now()

        if message == 'sitting_up':
            show_alert_window((0, 140, 255), 'Sitting')
            log_detection("sitting")
            print("偵測到坐起")

        elif message == 'turning_waist':
            show_alert_window((0, 0, 255), 'Turning')
            pygame.mixer.Sound.play(alarm_sound)
            log_detection("turning")
            print("偵測到轉腰")

        elif message == "lying_down":
            show_alert_window((0, 255, 0), 'Lying')
            log_detection("lying")
            print("偵測到平躺")

    except socket.timeout:
        # 如果超過指定時間未接收到任何訊息，檢查是否進入「無人狀態」
        if datetime.now() - last_detection_time > timedelta(seconds=2):
            show_alert_window((100, 100, 100), 'No Detection')
            log_detection("no_detection")
            print("無人偵測到")

    if cv.waitKey(1) == 27:  # 按下 ESC 鍵退出
        break

cv.destroyAllWindows()
pygame.quit()
