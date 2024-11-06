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
alarm_triggered = False  # 控制是否已經觸發警報
last_detection_time = time.time()  # 記錄最後一次檢測到人的時間
no_person_timeout = 5  # 設置超過 5 秒沒有檢測到人則觸發小警報

# 有效的後端和運行目標組合
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

# 解析命令列參數
parser = argparse.ArgumentParser(description='Pose Estimation from MediaPipe')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''選擇一個後端和目標組合來運行該演示:
                        {:d}: (默認) OpenCV + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
args = parser.parse_args()

# 可視化關鍵點和邊界框的函數
def visualize(image, poses):
    display_screen = image.copy()
    for idx, pose in enumerate(poses):
        bbox, landmarks_screen, landmarks_word, mask, heatmap, conf = pose

        # 繪製邊界框
        bbox = bbox.astype(np.int32)
        cv.rectangle(display_screen, bbox[0], bbox[1], (0, 255, 0), 2)
        cv.putText(display_screen, '{:.4f}'.format(conf), (bbox[0][0], bbox[0][1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # 繪製關鍵點
        keep_landmarks = landmarks_screen[:, 4] > 0.8  # 只顯示置信度超過0.8的關鍵點
        for i, p in enumerate(landmarks_screen[:, 0:2].astype(np.int32)):
            if keep_landmarks[i]:
                cv.circle(display_screen, p, 2, (0, 0, 255), -1)

    return display_screen

# 判斷是否躺下的函數
def is_lying_down(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    head_y = landmarks[0][1]  # 頭部
    return head_y > hip_y + 20  # 如果頭部比臀部低至少20像素，則判定為躺下

# 判斷是否起身的函數
def is_getting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return shoulder_y < hip_y - 20  # 如果肩膀比臀部高20像素以上，則判定為起身

# 判斷是否坐起的函數
def is_sitting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return hip_y > shoulder_y and shoulder_y > (hip_y - 20)  # 如果肩膀比臀部高但差距小於20像素，則判定為坐起

# 判斷是否扭腰的函數
def is_turning_waist(landmarks):
    left_shoulder_x = landmarks[11][0]  # 左肩膀
    right_shoulder_x = landmarks[12][0]  # 右肩膀
    hip_x = landmarks[23][0]  # 左臀部
    left_knee_y = landmarks[25][1]  # 左膝
    left_ankle_y = landmarks[27][1]  # 左腳踝

    # 確認肩膀水平，且臀部有水平偏移
    shoulders_aligned = abs(left_shoulder_x - right_shoulder_x) < 15
    hip_offset = abs(hip_x - left_shoulder_x) > 15
    # 確認腿部從垂直變為水平 (膝蓋和腳踝的 y 座標接近)
    legs_horizontal = abs(left_knee_y - left_ankle_y) < 20

    # 若肩膀對齊或腿部水平，並且臀部偏移，則判定為轉腰
    return (shoulders_aligned or legs_horizontal) and hip_offset

# 警報觸發函數
def trigger_alarm():
    print("!警報!")

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # 建立人檢測器
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
                                  nmsThreshold=0.3,
                                  scoreThreshold=0.5,
                                  topK=5000,
                                  backendId=backend_id,
                                  targetId=target_id)

    # 建立姿態估計器
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx',
                            confThreshold=0.8,
                            backendId=backend_id,
                            targetId=target_id)

    # 緩衝區，用來儲存最近的動作歷史
    history_buffer = deque(maxlen=3)  # 改成3以檢測連續動作

    # 開啟攝影機
    deviceId = 0
    cap = cv.VideoCapture(deviceId)

    tm = cv.TickMeter()
    alarm_triggered = False  # 用於控制是否已經觸發警報

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('無法獲取幀！')
            break

        # 檢測人
        persons = person_detector.infer(frame)
        poses = []

        tm.start()
        # 對每個檢測到的人進行姿態估計
        for person in persons:
            pose = pose_estimator.infer(frame, person)
            if pose is not None:
                poses.append(pose)
        tm.stop()

        # 繪製結果到影像上
        frame = visualize(frame, poses)

        if len(persons) > 0:
            last_detection_time = time.time()  # 更新最後偵測到人的時間

            for pose in poses:
                _, landmarks_screen, _, _, _, _ = pose

                # 根據姿勢判斷動作並記錄到緩衝區
                if is_lying_down(landmarks_screen):
                    movement = 'lying_down'
                elif is_sitting_up(landmarks_screen):
                    movement = 'sitting_up'
                elif is_turning_waist(landmarks_screen):
                    movement = 'turning_waist'
                elif is_getting_up(landmarks_screen):
                    movement = 'getting_up'
                else:
                    movement = 'unknown'

                # 終端顯示當前動作
                print(f'動作判斷: {movement}')

                # 每當動作變化，顯示當前動作
                cv.putText(frame, f'Current Movement: {movement}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 將動作加入緩衝區
                if len(history_buffer) == 0 or movement != history_buffer[-1]:
                    history_buffer.append(movement)

                # 根據緩衝區判斷連續動作是否符合警報條件
                if history_buffer.count('getting_up') == len(history_buffer):
                    trigger_alarm()
                    alarm_triggered = True
                elif history_buffer.count('sitting_up') == len(history_buffer):
                    # 如果偵測到使用者持續坐起，也可以觸發警報
                    trigger_alarm()
                    alarm_triggered = True

        # 如果超過指定時間沒有檢測到人，觸發小警報
        elif time.time() - last_detection_time > no_person_timeout:
            cv.putText(frame, 'No Person Detected!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 顯示FPS
        fps = tm.getFPS()
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 50), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

        # 顯示結果影像
        cv.imshow('MediaPipe Pose Detection Demo', frame)
        tm.reset()
