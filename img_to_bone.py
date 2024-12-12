import sys
import argparse
import numpy as np
import cv2 as cv

# 檢查 OpenCV 版本
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "請安裝較新的 opencv-python: python3 -m pip install --upgrade opencv-python"

from mp_pose_2 import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# 預設模型和檔案路徑
MODEL_PATH = './pose_estimation_mediapipe_2023mar.onnx'
INPUT_IMAGE = 'test_img.png'
OUTPUT_IMAGE = 'save.png'

# 用於視覺化的函式
def visualize(image, poses):
    display_screen = image.copy()

    def _draw_lines(image, landmarks, keep_landmarks, is_draw_point=True, thickness=5):
        def _draw_by_presence(idx1, idx2):
            if keep_landmarks[idx1] and keep_landmarks[idx2]:
                cv.line(image, landmarks[idx1], landmarks[idx2], (255, 255, 255), thickness)

        # 繪製人體骨架的連線
        _draw_by_presence(0, 1)
        _draw_by_presence(1, 2)
        _draw_by_presence(2, 3)
        _draw_by_presence(3, 7)
        _draw_by_presence(0, 4)
        _draw_by_presence(4, 5)
        _draw_by_presence(5, 6)
        _draw_by_presence(6, 8)

        _draw_by_presence(9, 10)

        _draw_by_presence(12, 14)
        _draw_by_presence(14, 16)
        _draw_by_presence(11, 13)
        _draw_by_presence(13, 15)

        # 繪製關節點
        if is_draw_point:
            for i, p in enumerate(landmarks):
                if keep_landmarks[i]:
                    cv.circle(image, p, thickness, (0, 0, 255), -1)

    for pose in poses:
        bbox, landmarks_screen, landmarks_word, mask, heatmap, conf = pose

        # 繪製關節點和骨架
        keep_landmarks = landmarks_screen[:, 4] > 0.8  # 只顯示可信度大於0.8的關節點
        landmarks_xy = landmarks_screen[:, 0:2].astype(np.int32)
        _draw_lines(display_screen, landmarks_xy, keep_landmarks)

    return display_screen

if __name__ == '__main__':
    # 設定 backend 和 target
    backend_id = cv.dnn.DNN_BACKEND_OPENCV
    target_id = cv.dnn.DNN_TARGET_CPU

    # 初始化偵測模型
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
                                  nmsThreshold=0.3,
                                  scoreThreshold=0.5,
                                  topK=5000,  # 通常只需一個人的偵測效果最好
                                  backendId=backend_id,
                                  targetId=target_id)

    pose_estimator = MPPose(modelPath=MODEL_PATH,
                            confThreshold=0.8,
                            backendId=backend_id,
                            targetId=target_id)

    # 載入圖片
    image = cv.imread(INPUT_IMAGE)
    if image is None:
        print(f"無法讀取圖片: {INPUT_IMAGE}")
        sys.exit(1)

    # 人員偵測
    persons = person_detector.infer(image)
    poses = []

    # 偵測每個人的姿態
    for person in persons:
        pose = pose_estimator.infer(image, person)
        if pose is not None:
            poses.append(pose)

    if len(persons) == 0:
        print("未偵測到任何人！")
    else:
        print("已偵測到人員！")

    # 可視化結果並儲存
    result_image = visualize(image, poses)
    cv.imwrite(OUTPUT_IMAGE, result_image)
    print(f"結果已儲存為 {OUTPUT_IMAGE}")
