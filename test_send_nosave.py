import sys
import numpy as np
import cv2 as cv
import socket
import time
from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 設定多播參數
MULTICAST_GROUP = '224.1.1.1'  # 任選私有多播地址
MULTICAST_PORT = 5005

def is_lying_down(landmarks):
    return landmarks[0][1] > landmarks[23][1] + 20

def is_sitting_up(landmarks):
    return landmarks[11][1] < landmarks[23][1] - 20

def is_turning_waist(landmarks):
    knee_hip_diff_left = abs(landmarks[25][1] - landmarks[23][1])
    knee_hip_diff_right = abs(landmarks[26][1] - landmarks[24][1])
    return knee_hip_diff_left < 30 and knee_hip_diff_right < 30

if __name__ == '__main__':
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx')
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx')

    cap = cv.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        persons = person_detector.infer(frame)
        if persons is not None and persons.size > 0:
            for person in persons:
                pose = pose_estimator.infer(frame, person)
                if pose:
                    _, landmarks_screen, _, _, _, _ = pose
                    movement = None

                    if is_turning_waist(landmarks_screen):
                        movement = 'turning_waist'
                    elif is_sitting_up(landmarks_screen):
                        movement = 'sitting_up'
                    elif is_lying_down(landmarks_screen):
                        movement = 'lying_down'

                    if movement:
                        sock.sendto(movement.encode(), (MULTICAST_GROUP, MULTICAST_PORT))
                        print(f"Current Movement: {movement}")

    cap.release()
    cv.destroyAllWindows()
