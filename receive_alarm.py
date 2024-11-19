import cv2 as cv
import numpy as np
import socket
import pygame

# 設定多播參數
MULTICAST_GROUP = '224.0.0.1'
MULTICAST_PORT = 4003

pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

def show_alert_window(color, text):
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
    data, _ = sock.recvfrom(1024)
    message = data.decode()

    if message == 'sitting_up':
        show_alert_window((0, 140, 255), '坐起')
        print("偵測到坐起")

    elif message == 'turning_waist':
        show_alert_window((0, 0, 255), '轉腰 - 警報!')
        pygame.mixer.Sound.play(alarm_sound)
        print("偵測到轉腰")

    elif message == "lying_down":
        show_alert_window((0, 255, 0), '平躺')
        print("偵測到平躺")

    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
pygame.quit()