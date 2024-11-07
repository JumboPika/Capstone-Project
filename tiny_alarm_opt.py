import sys
import argparse
import numpy as np
import cv2 as cv
from collections import deque
import time

from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 初始化變數
alarm_triggered = False
last_detection_time = time.time()
no_person_timeout = 5  # 超過5秒沒有檢測到人則觸發小警報

# 有效的後端和運行目標組合 (用字典結構方便查找)
backend_target_pairs = {
    0: (cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU),
    1: (cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA),
    2: (cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16),
    3: (cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU),
    4: (cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU)
}

# 解析命令列參數
parser = argparse.ArgumentParser(description='Pose Estimation from MediaPipe')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''選擇一個後端和目標組合來運行該演示:
                        0: (默認) OpenCV + CPU,
                        1: CUDA + GPU (CUDA),
                        2: CUDA + GPU (CUDA FP16),
                        3: TIM-VX + NPU,
                        4: CANN + NPU
                    ''')
args = parser.parse_args()

# 繪製邊界框和關鍵點
def visualize(image, poses):
    for pose in poses:
        bbox, landmarks, _, _, _, conf = pose
        bbox = bbox.astype(np.int32)
        cv.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)
        cv.putText(image, '{:.4f}'.format(conf), (bbox[0][0], bbox[0][1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # 顯示置信度高於0.8的關鍵點
        keep_landmarks = landmarks[:, 4] > 0.8
        for i, p in enumerate(landmarks[:, 0:2].astype(np.int32)):
            if keep_landmarks[i]:
                cv.circle(image, p, 2, (0, 0, 255), -1)
    return image

# 動作檢測函數
def detect_movement(landmarks):
    hip_y = landmarks[23][1]
    head_y = landmarks[0][1]
    shoulder_y = landmarks[11][1]
    knee_y = landmarks[25][1]
    ankle_y = landmarks[27][1]
    left_shoulder_x = landmarks[11][0]
    right_shoulder_x = landmarks[12][0]
    hip_x = landmarks[23][0]
    
    if head_y > hip_y + 20:
        return 'lying_down'
    elif shoulder_y < hip_y - 20:
        return 'sitting_up'
    elif hip_y < knee_y < ankle_y:
        return 'standing_up'
    elif abs(left_shoulder_x - right_shoulder_x) < 20 and abs(hip_x - left_shoulder_x) > 10:
        return 'turning_waist'
    return 'unknown'

# 警報觸發函數
def trigger_alarm():
    print("!警報!")

if __name__ == '__main__':
    backend_id, target_id = backend_target_pairs[args.backend_target]

    # 建立人檢測器和姿態估計器
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
                                  nmsThreshold=0.3,
                                  scoreThreshold=0.5,
                                  topK=5000,
                                  backendId=backend_id,
                                  targetId=target_id)

    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx',
                            confThreshold=0.8,
                            backendId=backend_id,
                            targetId=target_id)

    history_buffer = deque(maxlen=3)
    cap = cv.VideoCapture(0)
    tm = cv.TickMeter()
    alarm_triggered = False

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('無法獲取幀！')
            break

        # 人檢測
        persons = person_detector.infer(frame)
        poses = []

        tm.start()
        # 姿態估計
        for person in persons:
            pose = pose_estimator.infer(frame, person)
            if pose is not None:
                poses.append(pose)
        tm.stop()

        frame = visualize(frame, poses)

        if len(persons) > 0:
            last_detection_time = time.time()

            for pose in poses:
                _, landmarks, _, _, _, _ = pose
                movement = detect_movement(landmarks)

                # 終端顯示當前動作
                print(f'Current Movement: {movement}')

                # 動作變化時更新緩衝區
                if not history_buffer or movement != history_buffer[-1]:
                    history_buffer.append(movement)

                # 轉腰動作直接觸發警報
                if len(history_buffer) >= 2 and history_buffer[-2] == 'sitting_up' and history_buffer[-1] == 'turning_waist':
                    trigger_alarm()
                    alarm_triggered = True

                # 檢查緩衝區是否符合警報條件
                if len(set(history_buffer)) == 1 and history_buffer[0] in ('sitting_up', 'standing_up'):
                    trigger_alarm()
                    alarm_triggered = True

        # 若超過指定時間無人則觸發小警報
        elif time.time() - last_detection_time > no_person_timeout and not alarm_triggered:
            cv.putText(frame, "No person detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            trigger_alarm()
            alarm_triggered = True

        # 隔幾幀顯示 FPS 資訊，減少顯示開銷
        if int(tm.getCounter()) % 10 == 0:
            fps_info = 'FPS: {:.2f}'.format(tm.getFPS())
            cv.putText(frame, fps_info, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv.imshow('MediaPipe Pose Estimation Demo', frame)
        tm.reset()
