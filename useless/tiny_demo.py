import sys
import argparse

import numpy as np
import cv2 as cv

from mp_pose import MPPose

sys.path.append('../person_detection_mediapipe')
from mp_persondet import MPPersonDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

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

def visualize(image, poses):
    display_screen = image.copy()
    for idx, pose in enumerate(poses):
        bbox, landmarks_screen, landmarks_word, mask, heatmap, conf = pose

        # Draw box
        bbox = bbox.astype(np.int32)
        cv.rectangle(display_screen, bbox[0], bbox[1], (0, 255, 0), 2)
        cv.putText(display_screen, '{:.4f}'.format(conf), (bbox[0][0], bbox[0][1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # Draw key points
        keep_landmarks = landmarks_screen[:, 4] > 0.8
        for i, p in enumerate(landmarks_screen[:, 0:2].astype(np.int32)):
            if keep_landmarks[i]:
                cv.circle(display_screen, p, 2, (0, 0, 255), -1)

    return display_screen

def is_lying_down(landmarks):
    hip_y = landmarks[11][1]
    head_y = landmarks[0][1]
    return head_y > hip_y + 20

def is_getting_up(landmarks):
    hip_y = landmarks[11][1]
    shoulder_y = landmarks[5][1]
    return shoulder_y < hip_y - 20

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # person detector
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
                                  nmsThreshold=0.3,
                                  scoreThreshold=0.5,
                                  topK=5000,
                                  backendId=backend_id,
                                  targetId=target_id)

    # pose estimator
    pose_estimator = MPPose(modelPath='./pose_estimation_mediapipe_2023mar.onnx',
                            confThreshold=0.8,
                            backendId=backend_id,
                            targetId=target_id)

    # Omit input to call default camera
    deviceId = 0
    cap = cv.VideoCapture(deviceId)

    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # person detector inference
        persons = person_detector.infer(frame)
        poses = []

        tm.start()
        # Estimate the pose of each person
        for person in persons:
            pose = pose_estimator.infer(frame, person)
            if pose is not None:
                poses.append(pose)
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, poses)

        if len(persons) == 0:
            print('No person detected!')
        else:
            print('Person detected!')
            for pose in poses:
                _, landmarks_screen, _, _, _, _ = pose

                # 檢測動作
                if is_lying_down(landmarks_screen):
                    cv.putText(frame, 'Lying Down', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif is_getting_up(landmarks_screen):
                    cv.putText(frame, 'Getting Up', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display FPS
            cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (6, 15), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))

        cv.imshow('MediaPipe Pose Detection Demo', frame)
        tm.reset()
