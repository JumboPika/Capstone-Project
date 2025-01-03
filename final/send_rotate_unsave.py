# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2 as cv
import socket
from datetime import datetime
from mp_pose_2 import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 多播設定
MOVEMENT_MULTICAST_GROUP = '224.0.0.1'  # 動作多播地址
MOVEMENT_MULTICAST_PORT = 4003
VIDEO_MULTICAST_GROUP = '224.0.0.2'  # 視訊多播地址
VIDEO_MULTICAST_PORT = 5004

# 緩衝區大小
BUFFER_SIZE = 15  # 統一設定緩衝區大小


# 動作檢測函數
def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1]


def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1]


def is_turning_waist(landmarks):
    shoulder_diff = abs(landmarks[11][0] - landmarks[12][0])
    return shoulder_diff < 10


def is_stable_no_person(buffer):
    """檢查無人偵測的穩定狀態（允許少量人干擾）"""
    no_person_count = buffer.count(False)
    return no_person_count > BUFFER_SIZE * 0.8  # 超過 80% 判定為無人


def is_stable_movement(buffer):
    """檢查特定動作的穩定狀態（動作占比超過閾值）"""
    movement_counts = {m: buffer.count(m) for m in set(buffer) if m}
    dominant_movement, count = max(movement_counts.items(), key=lambda x: x[1], default=(None, 0))
    return dominant_movement, count > BUFFER_SIZE // 2  # 超過一半為穩定動作


if __name__ == '__main__':
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)

    # 降低解析度 (例如 640x480)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 15)  #15fps

    movement_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    movement_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    video_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    person_detected_buffer = [False] * BUFFER_SIZE
    movement_buffer = [None] * BUFFER_SIZE
    recording = False
    video_writer = None
    last_movement = None

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # 保留攝影機旋轉邏輯（順時針旋轉）
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        # 調整影像尺寸
        frame = cv.resize(frame, (240, 320))

        # 進行人物偵測
        persons = person_detector.infer(frame)
        person_detected = persons is not None and persons.size > 0

        # 更新 person_detected_buffer
        person_detected_buffer.append(person_detected)
        if len(person_detected_buffer) > BUFFER_SIZE:
            person_detected_buffer.pop(0)

        # 檢查穩定無人狀態
        stable_no_person = is_stable_no_person(person_detected_buffer)

        movement = None
        if person_detected:  # 有偵測到人
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose

                    # 檢測動作
                    if is_lying_down(landmarks_screen):
                        movement = 'lying_down'
                    elif is_turning_waist(landmarks_screen):
                        movement = 'turning_waist'
                    elif is_sitting_up(landmarks_screen):
                        movement = 'sitting_up'

                    # 更新 movement_buffer
                    movement_buffer.append(movement)
                    if len(movement_buffer) > BUFFER_SIZE:
                        movement_buffer.pop(0)

                    # 穩定動作判定
                    stable_movement, is_stable = is_stable_movement(movement_buffer)
                    if is_stable:
                        movement_sock.sendto(stable_movement.encode(),
                                             (MOVEMENT_MULTICAST_GROUP, MOVEMENT_MULTICAST_PORT))
                        print("Stable Movement: {}".format(stable_movement))

                    # 控制錄影邏輯（如果需要）
                    if movement == 'sitting_up' and not recording:
                        video_path = os.path.join('./example_outputs', "{}.mp4".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv.VideoWriter(video_path, fourcc, 15.0, (240, 320))
                        recording = True

                    if recording:
                        video_writer.write(frame)

                    if movement == 'lying_down' and last_movement == 'sitting_up' and recording:
                        recording = False
                        video_writer.release()

                    last_movement = movement

        elif stable_no_person:  # 穩定地沒偵測到人
            if recording:
                recording = False
                video_writer.release()

        # 壓縮並發送影像資料
        _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])
        max_packet_size = 1400
        data = buffer.tobytes()
        for i in range(0, len(data), max_packet_size):
            video_sock.sendto(data[i:i + max_packet_size], (VIDEO_MULTICAST_GROUP, VIDEO_MULTICAST_PORT))

    cap.release()
    if recording and video_writer:
        video_writer.release()
