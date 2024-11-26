import sys
import os
import numpy as np
import cv2 as cv
import socket
import time
from datetime import datetime
from mp_pose_2 import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 多播設定
MOVEMENT_MULTICAST_GROUP = '224.0.0.1'  # 動作多播地址
MOVEMENT_MULTICAST_PORT = 4003
VIDEO_MULTICAST_GROUP = '224.0.0.2'    # 視訊多播地址
VIDEO_MULTICAST_PORT = 5004

# 判斷是否躺下
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20

# 判斷是否坐起
def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20

def is_turning_waist(landmarks):
    shoulder_width_diff = abs(landmarks[11][0] - landmarks[12][0])  # 兩肩的水平距離
    return shoulder_width_diff < 10

# 初始化變數
last_movement = None
previous_sitting_up = False  # 新增變數來紀錄是否上次是坐起

if __name__ == '__main__':
    # 初始化模型與攝像頭
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # 動作多播套接字
    movement_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    movement_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    # 視訊多播套接字
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    video_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.resize(frame, (240, 320))

        # 進行人物偵測
        persons = person_detector.infer(frame)
        person_detected = persons is not None and persons.size > 0

        if person_detected:
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose
                    movement = None

                    # 判斷當前是躺下還是坐起
                    if is_lying_down(landmarks_screen):
                        movement = 'lying_down'
                        previous_sitting_up = False  # 更新為非坐起狀態
                    elif is_sitting_up(landmarks_screen):
                        if previous_sitting_up and is_turning_waist(landmarks_screen):
                            movement = 'turning_waist'
                        else:
                            movement = 'sitting_up'
                            previous_sitting_up = True  # 記錄坐起狀態

                    last_movement = movement

                    # 發送動作多播訊息
                    if movement:
                        movement_sock.sendto(movement.encode(), (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
                        print(f"Current Movement: {movement}")

        # 編碼影像並進行視訊多播
        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 20])
        max_packet_size = 1400
        data = buffer.tobytes()
        for i in range(0, len(data), max_packet_size):
            video_sock.sendto(data[i:i+max_packet_size], (VIDEO_MULTICAST_GROUP, VIDEO_MULTICAST_PORT))

        time.sleep(0.1)

    cap.release()
    cv.destroyAllWindows()
    movement_sock.close()
    video_sock.close()
