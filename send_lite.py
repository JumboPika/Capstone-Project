import sys
import os
import numpy as np
import cv2 as cv
import socket
import struct
import time
from datetime import datetime
from mp_pose_2 import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 多播設定
MOVEMENT_MULTICAST_GROUP = '224.0.0.1'
MOVEMENT_MULTICAST_PORT = 4003
VIDEO_MULTICAST_GROUP = '224.0.0.2'
VIDEO_MULTICAST_PORT = 5004

# 動作檢測函數
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20

def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20

def is_turning_waist(landmarks):
    knee_hip_diff_left = abs(landmarks[25][1] - landmarks[23][1])
    knee_hip_diff_right = abs(landmarks[26][1] - landmarks[24][1])
    return knee_hip_diff_left < 30 and knee_hip_diff_right < 30

# 初始化變數
recording = False
video_writer = None
video_path = None
last_movement = None
movement_buffer = []  # 緩衝區
BUFFER_SIZE = 5  # 緩衝區大小

def get_stable_movement(buffer):
    """根據緩衝區返回穩定的動作"""
    if len(buffer) < BUFFER_SIZE:
        return None
    return max(set(buffer), key=buffer.count)  # 返回出現次數最多的狀態

if __name__ == '__main__':
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    movement_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    movement_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    video_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    output_dir = './example_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame = cv.resize(frame, (320, 240))
        persons = person_detector.infer(frame)
        current_movement = None

        if persons is not None and persons.size > 0:
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose

                    if is_lying_down(landmarks_screen):
                        current_movement = 'lying_down'
                    elif is_sitting_up(landmarks_screen):
                        current_movement = 'sitting_up'
                    elif is_turning_waist(landmarks_screen):
                        current_movement = 'turning_waist'

        # 更新緩衝區並獲取穩定動作
        if current_movement:
            movement_buffer.append(current_movement)
            if len(movement_buffer) > BUFFER_SIZE:
                movement_buffer.pop(0)
            stable_movement = get_stable_movement(movement_buffer)
        else:
            stable_movement = None

        # 動作判斷與處理
        if stable_movement:
            if stable_movement == 'sitting_up' and not recording:
                print("開始錄影")
                recording = True
                last_movement = stable_movement
                video_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (320, 240))

            if recording:
                video_writer.write(frame)

            if stable_movement == 'turning_waist' and recording:
                print("轉身動作，保存影片")
                recording = False
                video_writer.release()

            if stable_movement == 'lying_down' and last_movement == 'sitting_up' and recording:
                print("坐起後躺下，丟棄影片")
                recording = False
                video_writer.release()
                os.remove(video_path)

            last_movement = stable_movement
            movement_sock.sendto(stable_movement.encode(), (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
            print(f"Current Movement: {stable_movement}")

        else:
            if recording and last_movement == 'sitting_up':
                print("坐起後未偵測到人，保存影片")
                recording = False
                video_writer.release()

            last_movement = None

        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])
        max_packet_size = 1400
        data = buffer.tobytes()
        for i in range(0, len(data), max_packet_size):
            video_sock.sendto(data[i:i+max_packet_size], (VIDEO_MULTICAST_GROUP, VIDEO_MULTICAST_PORT))

        time.sleep(0.1)

    cap.release()
    if recording and video_writer:
        video_writer.release()
    cv.destroyAllWindows()
    movement_sock.close()
    video_sock.close()
