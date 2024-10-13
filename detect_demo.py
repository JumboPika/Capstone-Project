import sys
import argparse
import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

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
parser.add_argument('--input', '-i', type=str,
                    help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./pose_estimation_mediapipe_2023mar.onnx',
                    help='Path to the model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.8,
                    help='Filter out hands of confidence < conf_threshold.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Specify to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, poses):
    display_screen = image.copy()
    display_3d = np.zeros((400, 400, 3), np.uint8)
    cv.line(display_3d, (200, 0), (200, 400), (255, 255, 255), 2)
    cv.line(display_3d, (0, 200), (400, 200), (255, 255, 255), 2)
    cv.putText(display_3d, 'Main View', (0, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Top View', (200, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Left View', (0, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Right View', (200, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    is_draw = False  # ensure only one person is drawn

    def _draw_lines(image, landmarks, keep_landmarks, is_draw_point=True, thickness=2):
        def _draw_by_presence(idx1, idx2):
            if keep_landmarks[idx1] and keep_landmarks[idx2]:
                cv.line(image, landmarks[idx1], landmarks[idx2], (255, 255, 255), thickness)

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
        _draw_by_presence(16, 22)
        _draw_by_presence(16, 18)
        _draw_by_presence(16, 20)
        _draw_by_presence(18, 20)

        _draw_by_presence(11, 13)
        _draw_by_presence(13, 15)
        _draw_by_presence(15, 21)
        _draw_by_presence(15, 19)
        _draw_by_presence(15, 17)
        _draw_by_presence(17, 19)

        _draw_by_presence(11, 12)
        _draw_by_presence(11, 23)
        _draw_by_presence(23, 24)
        _draw_by_presence(24, 12)

        _draw_by_presence(24, 26)
        _draw_by_presence(26, 28)
        _draw_by_presence(28, 30)
        _draw_by_presence(28, 32)
        _draw_by_presence(30, 32)

        _draw_by_presence(23, 25)
        _draw_by_presence(25, 27)
        _draw_by_presence(27, 31)
        _draw_by_presence(27, 29)
        _draw_by_presence(29, 31)

        if is_draw_point:
            for i, p in enumerate(landmarks):
                if keep_landmarks[i]:
                    cv.circle(image, p, thickness, (0, 0, 255), -1)

    for idx, pose in enumerate(poses):
        bbox, landmarks_screen, landmarks_word, mask, heatmap, conf = pose

        edges = cv.Canny(mask, 100, 200)
        kernel = np.ones((2, 2), np.uint8) # expansion edge to 2 pixels
        edges = cv.dilate(edges, kernel, iterations=1)
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        edges_bgr[edges == 255] = [0, 255, 0]
        display_screen = cv.add(edges_bgr, display_screen)

        # draw box
        bbox = bbox.astype(np.int32)
        cv.rectangle(display_screen, bbox[0], bbox[1], (0, 255, 0), 2)
        cv.putText(display_screen, '{:.4f}'.format(conf), (bbox[0][0], bbox[0][1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        # Draw line between each key points
        landmarks_screen = landmarks_screen[:-6, :]
        landmarks_word = landmarks_word[:-6, :]

        keep_landmarks = landmarks_screen[:, 4] > 0.8 # only show visible keypoints which presence bigger than 0.8

        landmarks_xy = landmarks_screen[:, 0: 2].astype(np.int32)
        _draw_lines(display_screen, landmarks_xy, keep_landmarks, is_draw_point=False)

        # z value is relative to HIP, but we use constant to instead
        for i, p in enumerate(landmarks_screen[:, 0: 3].astype(np.int32)):
            if keep_landmarks[i]:
                cv.circle(display_screen, np.array([p[0], p[1]]), 2, (0, 0, 255), -1)

        if is_draw is False:
            is_draw = True
            # Main view
            landmarks_xy = landmarks_word[:, [0, 1]]
            landmarks_xy = (landmarks_xy * 100 + 100).astype(np.int32)
            _draw_lines(display_3d, landmarks_xy, keep_landmarks, thickness=2)

            # Top view
            landmarks_xz = landmarks_word[:, [0, 2]]
            landmarks_xz[:, 1] = -landmarks_xz[:, 1]
            landmarks_xz = (landmarks_xz * 100 + np.array([300, 100])).astype(np.int32)
            _draw_lines(display_3d, landmarks_xz, keep_landmarks, thickness=2)

            # Left view
            landmarks_yz = landmarks_word[:, [2, 1]]
            landmarks_yz[:, 0] = -landmarks_yz[:, 0]
            landmarks_yz = (landmarks_yz * 100 + np.array([100, 300])).astype(np.int32)
            _draw_lines(display_3d, landmarks_yz, keep_landmarks, thickness=2)

            # Add conditions for lying down and getting up actions
            if check_lying_down(landmarks_word):
                cv.putText(display_screen, "Lying Down!", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif check_getting_up(landmarks_word):
                cv.putText(display_screen, "Getting Up!", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if args.vis:
        cv.imshow('Pose Estimation', display_screen)

def check_lying_down(landmarks):
    # A simple check: if the hip height is low compared to the head
    hip_height = landmarks[11][1]  # Use left hip as reference
    head_height = landmarks[0][1]   # Use head as reference
    return hip_height > head_height + 50  # Adjust threshold as needed

def check_getting_up(landmarks):
    # A simple check: if the hip is moving up from lying down position
    hip_height = landmarks[11][1]
    head_height = landmarks[0][1]
    return hip_height < head_height - 50  # Adjust threshold as needed

if __name__ == "__main__":
    # Initialize Person Detection
    person_det = MPPersonDet(args.backend_target)

    # Initialize Pose Estimation
    pose_estimator = MPPose(args.model, args.backend_target)

    if args.input:
        image = cv.imread(args.input)
        if image is None:
            print('Error loading image:', args.input)
            exit(-1)

        pose_estimates = pose_estimator.infer(image)
        visualize(image, pose_estimates)

    else:
        cap = cv.VideoCapture(0)
        while True:
            ret, image = cap.read()
            if not ret:
                print('Error reading video stream')
                break

            person_bboxes = person_det.infer(image)
            pose_estimates = []
            for bbox in person_bboxes:
                pose_estimates.append(pose_estimator.infer(image, bbox))

            visualize(image, pose_estimates)

            if cv.waitKey(1) == 27:  # Exit on ESC key
                break

        cap.release()

    if args.save:
        cv.imwrite('output.png', image)
