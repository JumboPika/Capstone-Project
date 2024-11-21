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

# 緩衝區判斷
def is_stable_no_person(buffer):
    return buffer.count(True) < 2  # 根據緩衝區判定是否穩定地沒偵測到人

# 初始化變數
recording = False
video_writer = None
video_path = None
last_movement = None

# 緩衝區大小
PERSON_BUFFER_SIZE = 5  # 可調整緩衝區大小
person_detected_buffer = [False] * PERSON_BUFFER_SIZE  # 初始為全「沒偵測到人」

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

        # 進行影像翻轉 (垂直翻轉或水平翻轉)
        #frame = cv.flip(frame, 0)  # `0` 為垂直翻轉，`1` 為水平翻轉，`-1` 為垂直+水平翻轉
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)  # 順時針旋轉

        # 後續邏輯 (例如尺寸調整、動作偵測)
        #frame = cv.resize(frame, (320, 240))
        frame = cv.resize(frame, (240, 320))

        # 進行人物偵測
        persons = person_detector.infer(frame)
        person_detected = persons is not None and persons.size > 0  # 判斷當前幀是否偵測到人

        # 更新「是否偵測到人」的緩衝區
        person_detected_buffer.append(person_detected)
        if len(person_detected_buffer) > PERSON_BUFFER_SIZE:
            person_detected_buffer.pop(0)  # 保持緩衝區固定大小

        # 判定是否穩定的沒偵測到人
        stable_no_person = is_stable_no_person(person_detected_buffer)

        if person_detected:  # 有偵測到人
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose
                    movement = None

                    # 檢測動作
                    if is_lying_down(landmarks_screen):
                        movement = 'lying_down'
                    elif is_sitting_up(landmarks_screen):
                        movement = 'sitting_up'
                    elif is_turning_waist(landmarks_screen):
                        movement = 'turning_waist'

                    # 開始錄影 (坐起)
                    if movement == 'sitting_up' and not recording:
                        print("開始錄影")
                        recording = True
                        last_movement = movement
                        video_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        #video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (320, 240))  # 降低幀率與大小
                        video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (240, 320))  # 降低幀率與大小

                    # 錄影過程
                    if recording:
                        video_writer.write(frame)

                    # 丟棄影片 (坐起 -> 躺下)
                    if movement == 'lying_down' and last_movement == 'sitting_up' and recording:
                        print("坐起後躺下，結束錄影並丟棄影片")
                        recording = False
                        video_writer.release()
                        os.remove(video_path)

                    # 更新最後動作
                    last_movement = movement

                    # 發送動作多播訊息
                    if movement:
                        movement_sock.sendto(movement.encode(), (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
                        print(f"Current Movement: {movement}")

        elif stable_no_person:  # 穩定地沒偵測到人
            if recording:
                print("坐起後未偵測到人，保存影片")
                recording = False
                video_writer.release()

            last_movement = None  # 清除最後動作記錄

        # 編碼影像為 JPEG 格式並進行視訊多播
        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])  # 壓縮影像品質
        max_packet_size = 1400
        data = buffer.tobytes()
        for i in range(0, len(data), max_packet_size):
            video_sock.sendto(data[i:i+max_packet_size], (VIDEO_MULTICAST_GROUP, VIDEO_MULTICAST_PORT))

        time.sleep(0.1)  # 增加間隔，降低幀率約 10fps

    # 釋放資源
    cap.release()
    if recording and video_writer:
        video_writer.release()
    cv.destroyAllWindows()
    movement_sock.close()
    video_sock.close()
