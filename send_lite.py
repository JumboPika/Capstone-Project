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
MOVEMENT_MULTICAST_GROUP = '224.0.0.1'  # 動作多播地址
MOVEMENT_MULTICAST_PORT = 4003
VIDEO_MULTICAST_GROUP = '224.0.0.2'    # 視訊多播地址
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
last_detection_time = time.time()
cooldown_time = 2  # 動作偵測冷卻時間（秒）

if __name__ == '__main__':
    # 初始化模型與攝像頭
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)

    # 降低解析度 (例如 640x480)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # 動作多播套接字
    movement_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    movement_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    # 視訊多播套接字
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    video_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    # 確保輸出資料夾存在
    output_dir = './example_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # 調整影像大小以減少負載
        frame = cv.resize(frame, (320, 240))  # 調整大小為 320x240

        # 控制辨識頻率
        current_time = time.time()
        if current_time - last_detection_time < 0.5:  # 每 0.5 秒進行一次動作偵測
            continue

        last_detection_time = current_time  # 更新上次偵測時間

        # 進行人物偵測與姿勢估計
        persons = person_detector.infer(frame)
        if persons is not None and persons.size > 0:
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose
                    movement = None

                    # 檢測動作
                    if is_turning_waist(landmarks_screen):
                        movement = 'turning_waist'
                    elif is_sitting_up(landmarks_screen):
                        movement = 'sitting_up'
                    elif is_lying_down(landmarks_screen):
                        movement = 'lying_down'

                    # 冷卻時間內跳過重複警報
                    if time.time() - last_detection_time < cooldown_time:
                        movement = None

                    # 開始錄影 (坐起)
                    if movement == 'sitting_up' and not recording:
                        print("開始錄影")
                        recording = True
                        video_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (320, 240))  # 降低幀率與大小

                    # 錄影過程
                    if recording:
                        video_writer.write(frame)

                    # 停止錄影 (躺下)
                    if movement == 'lying_down' and recording:
                        print("結束錄影並丟棄影片")
                        recording = False
                        video_writer.release()
                        os.remove(video_path)  # 刪除未保存影片

                    # 保存影片 (轉身)
                    if movement == 'turning_waist' and recording:
                        print("轉身動作，保存影片")
                        recording = False
                        video_writer.release()

                    # 發送動作多播訊息
                    if movement:
                        movement_sock.sendto(movement.encode(), (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
                        print(f"Current Movement: {movement}")

        # 編碼影像為 JPEG 格式並進行視訊多播
        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])  # 壓縮影像品質
        max_packet_size = 1400
        data = buffer.tobytes()
        for i in range(0, len(data), max_packet_size):
            video_sock.sendto(data[i:i+max_packet_size], (VIDEO_MULTICAST_GROUP, VIDEO_MULTICAST_PORT))

        time.sleep(0.05)  # 增加間隔，降低幀率約 20fps

    # 釋放資源
    cap.release()
    if recording and video_writer:
        video_writer.release()
    cv.destroyAllWindows()
    movement_sock.close()
    video_sock.close()
