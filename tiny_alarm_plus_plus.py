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

# 動作判斷函數
def is_lying_down(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    head_y = landmarks[0][1]  # 頭部
    return head_y > hip_y + 20  # 如果頭部比臀部低至少20像素，則判定為躺下

def is_sitting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return shoulder_y < hip_y - 20 # 如果肩膀比臀部高，但差距小於20像素，則判定為坐起

def is_standing_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    knee_y = landmarks[25][1]  # 左膝
    ankle_y = landmarks[27][1]  # 左腳踝
    return hip_y < knee_y < ankle_y  # 如果臀部高於膝蓋，且膝蓋高於腳踝，則判定為站立

def is_turning_waist(landmarks):
    left_knee_y = landmarks[25][1]  # 左膝
    right_knee_y = landmarks[26][1]  # 右膝
    left_hip_y = landmarks[23][1]   # 左臀部
    right_hip_y = landmarks[24][1]  # 右臀部
    knee_hip_diff_left = abs(left_knee_y - left_hip_y)
    knee_hip_diff_right = abs(right_knee_y - right_hip_y)
    is_knee_level_left = knee_hip_diff_left < 30
    is_knee_level_right = knee_hip_diff_right < 30
    return is_knee_level_left and is_knee_level_right

# 新增：動作穩定性檢查，避免瞬間變動誤判
def is_stable_position(landmarks, prev_landmarks, threshold=10):
    if prev_landmarks is None:
        return True  # 如果沒有前一幀，則無法比較
    # 比較特定關節（例如左肩膀和左臀部）的 y 座標穩定性
    return (abs(landmarks[11][1] - prev_landmarks[11][1]) < threshold and  # 左肩膀
            abs(landmarks[23][1] - prev_landmarks[23][1]) < threshold)     # 左臀部

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
    history_buffer = deque(maxlen=5)  # 增大緩衝區以檢測穩定動作
    prev_landmarks = None  # 用於記錄上一幀的關節位置

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

                # 確認位置穩定後再進行動作判斷
                if is_stable_position(landmarks_screen, prev_landmarks):
                    if is_standing_up(landmarks_screen):
                        movement = 'standing_up'
                    elif is_turning_waist(landmarks_screen):
                        movement = 'turning_waist'
                    elif is_sitting_up(landmarks_screen):
                        movement = 'sitting_up'
                    elif is_lying_down(landmarks_screen):
                        movement = 'lying_down'
                    else:
                        movement = 'unknown'
                    
                    # 終端顯示當前動作
                    print(f'Current Movement: {movement}')

                    # 當動作有變化時才更新緩衝區
                    if len(history_buffer) == 0 or movement != history_buffer[-1]:
                        history_buffer.append(movement)

                    # 當檢測到「坐起」變成「轉腰」時立即觸發警報
                    if len(history_buffer) >= 2 and history_buffer[-2] == 'sitting_up' and history_buffer[-1] == 'turning_waist':
                        trigger_alarm()
                        alarm_triggered = True

                    # 判斷連續動作是否符合警報條件
                    if history_buffer.count('standing_up') > len(history_buffer) // 2:
                        trigger_alarm()
                        alarm_triggered = True
                    elif history_buffer.count('sitting_up') > len(history_buffer) // 2:
                        trigger_alarm()
                        alarm_triggered = True

                prev_landmarks = landmarks_screen  # 記錄當前的關節位置作為下一幀的參考

        # 如果超過指定時間沒有檢測到人，觸發小警報
        elif time.time() - last_detection_time > no_person_timeout:
            cv.putText(frame, "No person detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if not alarm_triggered:  # 如果警報尚未觸發
                trigger_alarm()
                alarm_triggered = True

        # 顯示處理後的幀
        cv.imshow('MediaPipe Pose Estimation Demo', frame)

        # 顯示 FPS 資訊
        fps_info = 'FPS: {:.2f}'.format(tm.getFPS())
        cv.putText(frame, fps_info, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        tm.reset()
