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

# 判斷是否躺下
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20

# 判斷是否坐起
def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20

def is_turning_waist(landmarks):
    """
    優化轉身判斷：臀部與肩膀角度變化較大，兩肩之間的水平距離差異較大
    """
    #knee_hip_diff_left = abs(landmarks[25][1] - landmarks[23][1])  # 左膝與臀部的距離
    #knee_hip_diff_right = abs(landmarks[26][1] - landmarks[24][1])  # 右膝與臀部的距離
    #knee_horizontal_diff = abs(landmarks[25][0] - landmarks[26][0])  # 左右膝的水平距離

    shoulder_width_diff = abs(landmarks[11][0] - landmarks[12][0])  # 兩肩的水平距離
    #shoulder_to_hip_left = abs(landmarks[11][1] - landmarks[23][1])  # 左肩與左臀部的垂直距離
    #shoulder_to_hip_right = abs(landmarks[12][1] - landmarks[24][1])  # 右肩與右臀部的垂直距離

    # 判斷為轉身：膝部距離較小，肩膀與臀部距離有變化，且肩膀間的水平距離大
    #return (knee_hip_diff_left < 30 and knee_hip_diff_right < 30 and knee_horizontal_diff > 50) and \
    #       (shoulder_width_diff > 30) and \
    #       (shoulder_to_hip_left > 20 or shoulder_to_hip_right > 20)
    return shoulder_width_diff < 10

# 初始化變數
recording = False
video_writer = None
video_path = None
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

    # 確保輸出資料夾存在
    output_dir = './example_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                        # 如果是躺下，直接更新狀態
                        movement = 'lying_down'
                        previous_sitting_up = False  # 更新為非坐起狀態
                    elif is_sitting_up(landmarks_screen):
                        # 如果是坐起，記錄狀態並進一步檢查轉身
                        if previous_sitting_up and is_turning_waist(landmarks_screen):
                            movement = 'turning_waist'
                        else:
                            movement = 'sitting_up'
                            previous_sitting_up = True  # 記錄坐起狀態


                    # 開始錄影
                    if movement == 'sitting_up' and not recording:
                        print("開始錄影")
                        recording = True
                        last_movement = movement
                        video_path = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (240, 320))

                    # 錄影過程
                    if recording:
                        video_writer.write(frame)

                    # (坐起 -> 躺下)
                    if movement == 'lying_down' and last_movement == 'sitting_up' and recording:
                        print("坐起後躺下，結束錄影")
                        recording = False
                        video_writer.release()

                    last_movement = movement

                    # 發送動作多播訊息
                    if movement:
                        movement_sock.sendto(movement.encode(), (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
                        print(f"Current Movement: {movement}")

        # 若穩定未偵測到人，結束錄影並儲存影片
        elif last_movement == 'sitting_up' and not person_detected:
            if recording:
                print("坐起後未偵測到人，保存影片")
                recording = False
                video_writer.release()

            last_movement = None

        # 編碼影像並進行視訊多播
        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 20])
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
