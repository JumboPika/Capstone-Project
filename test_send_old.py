# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2 as cv
import time
import socket
from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 設定多播參數
MULTICAST_GROUP = '224.1.1.1'  # 任選私有多播地址
MULTICAST_PORT = 5005

# 設定錄影參數
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
fourcc = cv.VideoWriter_fourcc(*'H264')
video_dir = './example_outputs'

recording = False
video_writer = None
video_filename = None

# 開始錄影
def start_recording(frame):
    global recording, video_writer, video_filename
    timestamp = int(time.time())
    # video_filename = f"{video_dir}/video_{timestamp}.avi"
    video_filename = f"{video_dir}/video_{timestamp}.mp4"
    video_writer = cv.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    recording = True
    print("開始錄影")

# 停止錄影並選擇是否保存影片
def stop_recording(save):
    global recording, video_writer, video_filename
    if recording:
        video_writer.release()
        if not save and video_filename:
            os.remove(video_filename)
            print("錄影已停止，影片已捨棄")
        elif save:
            print("錄影已停止，影片已保存")
        recording = False
        video_filename = None

# 判斷是否躺下
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20

# 判斷是否坐起來
def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20

# 判斷是否轉腰
def is_turning_waist(landmarks):
    knee_hip_diff_left = abs(landmarks[25][1] - landmarks[23][1])
    knee_hip_diff_right = abs(landmarks[26][1] - landmarks[24][1])
    return knee_hip_diff_left < 30 and knee_hip_diff_right < 30

if __name__ == '__main__':
    # 初始化人物檢測和姿勢估計模型
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    # 設定攝影機和多播UDP套接字
    cap = cv.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    # 進行影像處理和動作偵測
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # 偵測人物
        persons = person_detector.infer(frame)
        if persons is not None and persons.size > 0:
            for person in persons:
                # 姿勢估計
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose

                    # 根據姿勢檢測動作
                    if is_turning_waist(landmarks_screen):
                        if recording:
                            stop_recording(save=True)
                        movement = 'turning_waist'

                    elif is_sitting_up(landmarks_screen):
                        if not recording:
                            start_recording(frame)
                        movement = 'sitting_up'

                    elif is_lying_down(landmarks_screen):
                        if recording:
                            stop_recording(save=False)
                        movement = 'lying_down'
                    
                    # 只有在 movement 被賦值後才發送
                    if movement:
                        sock.sendto(movement.encode(), (MULTICAST_GROUP, MULTICAST_PORT))
                        print(f"Current Movement: {movement}")
    
    # 釋放資源
    cap.release()
    cv.destroyAllWindows()
