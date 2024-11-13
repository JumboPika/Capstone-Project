import sys
import argparse
import os
import numpy as np
import cv2 as cv
import time
import socket
from collections import deque
from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 初始化變數
recording = False
video_writer = None
alarm_triggered = False
last_detection_time = time.time()
no_person_timeout = 5
# 設定多播參數
MULTICAST_GROUP = '224.1.1.1'  # 任選私有多播地址
MULTICAST_PORT = 5005

# 設定錄影參數
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 選擇影片編碼格式
video_dir = '/example_outputs/'  # 設定影片存儲路徑
# 設置Socket傳輸
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
laptop_ip = '172.17.38.114'  # 替換為筆電的IP地址
laptop_port = 5005
# 判斷動作的函數
def is_lying_down(landmarks):
    hip_y = landmarks[23][1]
    head_y = landmarks[0][1]
    return head_y > hip_y + 20
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#fourcc = cv.VideoWriter_fourcc(*'MJPG')
fourcc = cv.VideoWriter_fourcc(*'H264')
video_dir = './example_outputs'

def is_sitting_up(landmarks):
    hip_y = landmarks[23][1]
    shoulder_y = landmarks[11][1]
    return shoulder_y < hip_y - 20
def is_turning_waist(landmarks):
    left_knee_y = landmarks[25][1]
    right_knee_y = landmarks[26][1]
    left_hip_y = landmarks[23][1]
    right_hip_y = landmarks[24][1]
    knee_hip_diff_left = abs(left_knee_y - left_hip_y)
    knee_hip_diff_right = abs(right_knee_y - right_hip_y)
    return knee_hip_diff_left < 30 and knee_hip_diff_right < 30
recording = False
video_writer = None
video_filename = None

# 開始錄影
def start_recording(frame):
    global recording, video_writer
    global recording, video_writer, video_filename
    timestamp = int(time.time())
    video_filename = f"{video_dir}/video_{timestamp}.avi"
    #video_filename = f"{video_dir}/video_{timestamp}.avi"
    video_filename = f"{video_dir}/video_{timestamp}.mp4"
    video_writer = cv.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    recording = True
    print("開始錄影")

# 停止錄影並保存影片
def stop_recording(save):
    global recording, video_writer
    global recording, video_writer, video_filename
    if recording:
        video_writer.release()
        if not save:
            video_filename = video_writer.filename
        if not save and video_filename:
            os.remove(video_filename)
            print("錄影已停止，影片已捨棄")
        else:
        elif save:
            print("錄影已停止，影片已保存")
        recording = False
        video_filename = None
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20
def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20
def is_turning_waist(landmarks):
    knee_hip_diff_left = abs(landmarks[25][1] - landmarks[23][1])
    knee_hip_diff_right = abs(landmarks[26][1] - landmarks[24][1])
    return knee_hip_diff_left < 30 and knee_hip_diff_right < 30

if __name__ == '__main__':
    # 建立人檢測器和姿態估計器
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # 偵測人並估計姿勢
        persons = person_detector.infer(frame)
        if persons is not None and len(persons) > 0:
        if persons is not None and persons.size > 0:
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose is None:
                    continue  # 若無效，跳過本次循環
                _, landmarks_screen, _, _, _, _ = pose
                # 動作判斷邏輯
                if is_sitting_up(landmarks_screen):
                    if not recording:
                        start_recording(frame)
                    movement = 'sitting_up'
                elif is_lying_down(landmarks_screen):
                    if recording:
                        stop_recording(save=False)  # 停止並捨棄錄影
                    movement = 'lying_down'
                elif is_turning_waist(landmarks_screen):
                    if recording:
                        stop_recording(save=True)  # 停止並保存錄影
                    movement = 'turning_waist'
                # 傳送狀態給筆電
                sock.sendto(movement.encode(), (laptop_ip, laptop_port))
                print(f"Current Movement: {movement}")
        if time.time() - last_detection_time > no_person_timeout:
            if not alarm_triggered:
                sock.sendto("No person detected".encode(), (laptop_ip, laptop_port))
                alarm_triggered = True
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose
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
                    sock.sendto(movement.encode(), (MULTICAST_GROUP, MULTICAST_PORT))
                    print(f"Current Movement: {movement}")
    cap.release()
    cv.destroyAllWindows()
