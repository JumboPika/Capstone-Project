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

# 判斷是否坐起的函數 (起身)
def is_sitting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return shoulder_y < hip_y - 20 # 如果肩膀比臀部高但差距小於20像素，則判定為坐起

# 判斷是否站立的函數
def is_standing_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    knee_y = landmarks[25][1]  # 左膝
    ankle_y = landmarks[27][1]  # 左腳踝

    # 如果臀部高於膝蓋，並且膝蓋高於腳踝，則判定為站立
    return hip_y < knee_y < ankle_y
'''
# 判斷是否轉腰的函數
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
'''
'''
def is_turning_waist(landmarks):
    left_shoulder_x = landmarks[11][0]  # 左肩膀
    right_shoulder_x = landmarks[12][0]  # 右肩膀
    hip_x = landmarks[23][0]  # 左臀部
    left_knee_y = landmarks[25][1]  # 左膝
    left_ankle_y = landmarks[27][1]  # 左腳踝

    # 寬鬆判定肩膀是否對齊
    shoulders_aligned = abs(left_shoulder_x - right_shoulder_x) < 20  # 擴大允許的偏差範圍到30

    # 寬鬆判定臀部是否有水平偏移
    hip_offset = abs(hip_x - left_shoulder_x) > 10  # 降低偏移門檻到10

    # 寬鬆判定腿部是否水平
    legs_horizontal = abs(left_knee_y - left_ankle_y) < 30  # 擴大允許的y軸差距到40

    # 若肩膀對齊或腿部水平，且臀部偏移，則判定為轉腰
    return (shoulders_aligned or legs_horizontal) and hip_offset
'''
def is_turning_waist(landmarks):
    # 獲取左膝、右膝和左臀部的 Y 座標
    left_knee_y = landmarks[25][1]  # 左膝
    right_knee_y = landmarks[26][1]  # 右膝
    left_hip_y = landmarks[23][1]   # 左臀部
    right_hip_y = landmarks[24][1]  # 右臀部

    # 計算膝蓋和臀部之間的垂直距離差
    knee_hip_diff_left = abs(left_knee_y - left_hip_y)
    knee_hip_diff_right = abs(right_knee_y - right_hip_y)

    # 判斷膝蓋和臀部是否接近水平（即差距小於一定的門檻）
    is_knee_level_left = knee_hip_diff_left < 30  # 你可以調整這個數值來改變靈敏度
    is_knee_level_right = knee_hip_diff_right < 30

    # 如果兩側膝蓋和臀部接近水平，則返回 True，表示轉腰或側身
    return is_knee_level_left and is_knee_level_right



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

                # 每當動作變化，顯示當前動作
                cv.putText(frame, f'Current Movement: {movement}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 將動作加入緩衝區
                if len(history_buffer) == 0 or movement != history_buffer[-1]:
                    history_buffer.append(movement)

                # 當檢測到坐姿變成轉腰時立即觸發警報
                if len(history_buffer) >= 2 and history_buffer[-2] == 'sitting_up' and history_buffer[-1] == 'turning_waist':
                    trigger_alarm()
                    alarm_triggered = True

                # 根據緩衝區判斷其他連續動作是否符合警報條件
                if history_buffer.count('getting_up') == len(history_buffer):
                    trigger_alarm()
                    alarm_triggered = True
                elif history_buffer.count('sitting_up') == len(history_buffer):
                    trigger_alarm()
                    alarm_triggered = True

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
