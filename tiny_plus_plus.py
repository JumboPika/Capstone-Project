import sys
import argparse
import numpy as np
import cv2 as cv
from collections import deque
from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 有效的後端與運行目標組合
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

# 命令行參數解析
parser = argparse.ArgumentParser(description='Pose Estimation from MediaPipe')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
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
        keep_landmarks = landmarks_screen[:, 4] > 0.8
        for i, p in enumerate(landmarks_screen[:, 0:2].astype(np.int32)):
            if keep_landmarks[i]:
                cv.circle(display_screen, p, 2, (0, 0, 255), -1)

    return display_screen

# 判斷是否躺下的函數（假設頭部位置比臀部低）
def is_lying_down(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    head_y = landmarks[0][1]  # 頭
    return head_y > hip_y + 20  # 頭部低於臀部，視為躺下

# 判斷是否起身的函數（肩膀上升到臀部以上）
def is_getting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return shoulder_y < hip_y - 20  # 肩膀高於臀部

# 判斷是否坐起的函數（肩膀與臀部位置相對）
def is_sitting_up(landmarks):
    hip_y = landmarks[23][1]  # 左臀部
    shoulder_y = landmarks[11][1]  # 左肩膀
    return hip_y > shoulder_y and shoulder_y > (hip_y - 20)  # 肩膀稍高於臀部

# 判斷是否扭腰的函數（根據肩膀與臀部位置判斷）
def is_turning_waist(landmarks):
    left_shoulder_x = landmarks[11][0]  # 左肩膀
    right_shoulder_x = landmarks[12][0]  # 右肩膀
    hip_x = landmarks[23][0]  # 左臀部
    return abs(left_shoulder_x - right_shoulder_x) < 15 and abs(hip_x - left_shoulder_x) > 15

# 主程式
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

    # 動作歷史緩衝區
    history_buffer = deque(maxlen=10)
    cap = cv.VideoCapture(0)

    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
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

        if len(persons) == 0:
            print('No person detected!')
        else:
            print('Person detected!')
            for pose in poses:
                _, landmarks_screen, _, _, _, _ = pose

                # 根據姿勢判斷動作並記錄到緩衝區
                if is_lying_down(landmarks_screen):
                    movement = 'lying_down'
                    cv.putText(frame, 'Lying Down', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif is_sitting_up(landmarks_screen):
                    movement = 'sitting_up'
                    cv.putText(frame, 'Sitting Up', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif is_turning_waist(landmarks_screen):
                    movement = 'turning_waist'
                    cv.putText(frame, 'Turning Waist', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif is_getting_up(landmarks_screen):
                    movement = 'getting_up'
                    cv.putText(frame, 'Getting Up', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    movement = 'unknown'

                # 將動作加入緩衝區
                history_buffer.append(movement)

                # 根據緩衝區的多數狀態來判斷最終狀態
                if history_buffer.count('lying_down') > len(history_buffer) // 2:
                    final_state = 'Lying Down'
                elif history_buffer.count('sitting_up') > len(history_buffer) // 2:
                    final_state = 'Sitting Up'
                elif history_buffer.count('turning_waist') > len(history_buffer) // 2:
                    final_state = 'Turning Waist'
                elif history_buffer.count('getting_up') > len(history_buffer) // 2:
                    final_state = 'Getting Up'
                else:
                    final_state = 'Unknown'

                # 顯示最終狀態
                cv.putText(frame, f'Final State: {final_state}', (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 顯示FPS
            cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (6, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # 顯示結果影像
        cv.imshow('MediaPipe Pose Detection Demo', frame)
        tm.reset()
