import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

jugetTime = 0

# 画图函数
def draws(image, post_landmarks, index):
    cv2.circle(image,
               (int(results.pose_landmarks.landmark[index].x * image.shape[1]),
                int(results.pose_landmarks.landmark[index].y * image.shape[0])),
               30, (255, 255, 255), 4)


def juget(image, post_landmarks):
    x11 = int(results.pose_landmarks.landmark[11].x * image.shape[1])
    y11 = int(results.pose_landmarks.landmark[11].y * image.shape[0])

    x12 = int(results.pose_landmarks.landmark[12].x * image.shape[1])
    y12 = int(results.pose_landmarks.landmark[12].y * image.shape[0])

    x23 = int(results.pose_landmarks.landmark[23].x * image.shape[1])
    y23 = int(results.pose_landmarks.landmark[23].y * image.shape[0])

    x24 = int(results.pose_landmarks.landmark[24].x * image.shape[1])
    y24 = int(results.pose_landmarks.landmark[24].y * image.shape[0])

    if y11 < y23 or y12 < y24 :
        drowningTime = time.time()
        if abs((x23-x11)/(y23-y11)) < 0.4 or abs((x24-x12)/(y24-y12)) < 0.4:  # 斜率绝对值小于 0.4 视为非正常泳姿

            image = cv2.putText(image, 'Drowning Warning!', (int(image.shape[1]*0.1), int(image.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.line(image,(x11,y11),(x12,y12),(0,0,255), 4)
            cv2.line(image, (x12,y12),(x24,y24), (0,0,255), 4)
            cv2.line(image, (x24,y24),(x23,y23), (0,0,255), 4)
            cv2.line(image, (x23,y23),(x11,y11), (0,0,255), 4)


    return image

if __name__ == '__main__':
    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    fps = 0.0
    cv2.namedWindow("MediaPipe Pose",flags=cv2.WINDOW_NORMAL)
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue



        t1 = time.time()
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # print("Num:",len(results.pose_landmarks.landmark))
        try: # 未检测到会抛出输出异常
            draws(image, results, 11)
            draws(image, results, 12)
            draws(image, results, 23)
            draws(image, results, 24)
            image = juget(image, results)
        except Exception as e:
            # print(e)
            pass
        # for i in range(len(results.pose_landmarks.landmark)):
        fps = (fps + (1. / (time.time() - t1))) / 2
        image = cv2.putText(image, 'FPS:'+str(int(fps)), (int(image.shape[1] * 0.1), int(image.shape[0] * 0.2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Pose', image)



        print("fps= %.2f" % (fps))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()