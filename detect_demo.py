import cv2 as cv
import numpy as np

# 偵測是否有躺下或起身的動作
def detect_pose_action(landmarks):
    """
    偵測是否有躺下或起身的動作
    :param landmarks: 2D 坐標的關鍵點位置
    :return: 動作名稱 ('lying_down', 'getting_up', 'standing', None)
    """
    shoulder_y = landmarks[11][1]  # 左肩膀
    hip_y = landmarks[23][1]       # 左臀部
    foot_y = landmarks[27][1]      # 左腳
    
    # 偵測躺下：當肩膀和臀部接近同一高度，且腳也在相同水平面時
    if abs(shoulder_y - hip_y) < 30 and abs(hip_y - foot_y) < 30:
        return 'lying_down'

    # 偵測起身：當肩膀從低處迅速上升時（例如：剛從躺著開始站起來）
    elif shoulder_y < hip_y - 50:  # 肩膀比臀部高很多，表示有起身的動作
        return 'getting_up'

    # 偵測站立：當肩膀比臀部高，但腳在底部
    elif shoulder_y < hip_y and foot_y > hip_y:
        return 'standing'

    return None

# 繪製和顯示人體姿勢
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
    detected_action = None

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
        kernel = np.ones((2, 2), np.uint8)  # expansion edge to 2 pixels
        edges = cv.dilate(edges, kernel, iterations=1)
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        edges_bgr[edges == 255] = [0, 255, 0]
        display_screen = cv.add(edges_bgr, display_screen)

        # draw box
        bbox = bbox.astype(np.int32)
        cv.rectangle(display_screen, bbox[0], bbox[1], (0, 255, 0), 2)
        cv.putText(display_screen, '{:.4f}'.format(conf), (bbox[0][0], bbox[0][1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        landmarks_screen = landmarks_screen[:-6, :]
        keep_landmarks = landmarks_screen[:, 4] > 0.8  # only show visible keypoints with confidence > 0.8

        landmarks_xy = landmarks_screen[:, 0:2].astype(np.int32)
        _draw_lines(display_screen, landmarks_xy, keep_landmarks, is_draw_point=False)

        if is_draw is False:
            is_draw = True

        # 偵測動作
        detected_action = detect_pose_action(landmarks_xy)

    return display_screen, display_3d, detected_action

# 主程式
if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # person detector
    person_detector = MPPersonDet(modelPath='../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
                                  nmsThreshold=0.3,
                                  scoreThreshold=0.5,
                                  topK=5000,  # usually only one person has good performance
                                  backendId=backend_id,
                                  targetId=target_id)
    # pose estimator
    pose_estimator = MPPose(modelPath=args.model,
                            confThreshold=args.conf_threshold,
                            backendId=backend_id,
                            targetId=target_id)

    # 如果輸入是圖片
    if args.input:
        image = cv.imread(args.input)
        persons = person_detector(image)
        poses = []
        for person in persons:
            pose = pose_estimator(image, person)
            if pose is not None:
                poses.append(pose)

        if len(poses) > 0:
            display_screen, display_3d, detected_action = visualize(image, poses)

            # 顯示動作檢測結果
            if detected_action == 'lying_down':
                cv.putText(display_screen, 'Warning: Lying down detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif detected_action == 'getting_up':
                cv.putText(display_screen, 'Warning: Getting up detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv.putText(display_screen, 'Person detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            display_screen = image.copy()
            cv.putText(display_screen, 'No person detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.save:
            cv.imwrite('output.png', display_screen)
        if args.vis:
            cv.imshow('Result', display_screen)
            cv.imshow('3D', display_3d)
            cv.waitKey(0)
    else:
        # 來自網路攝影機的即時檢測
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            persons = person_detector(frame)
            poses = []
            for person in persons:
                pose = pose_estimator(frame, person)
                if pose is not None:
                    poses.append(pose)

            if len(poses) > 0:
                display_screen, display_3d, detected_action = visualize(frame, poses)

                # 顯示動作檢測結果
                if detected_action == 'lying_down':
                    cv.putText(display_screen, 'Warning: Lying down detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif detected_action == 'getting_up':
                    cv.putText(display_screen, 'Warning: Getting up detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv.putText(display_screen, 'Person detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                display_screen = frame.copy()
                cv.putText(display_screen, 'No person detected', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow('Result', display_screen)
            cv.imshow('3D', display_3d)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
