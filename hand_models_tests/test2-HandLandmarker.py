import mediapipe as mp
import cv2
import datetime as dt
import socket

# Sockets
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort1 = ("127.0.0.1", 5052)
serverAddressPort2 = ("127.0.0.1", 5053)

# Gesture Recognition
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Camera
video = cv2.VideoCapture(1)

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('gesture recognition result: {}'.format(result))
    if result.hand_landmarks:
        data = [[],[]]
        # Iterate hands
        for i in range(len(result.hand_world_landmarks)):
            index = 1 - result.handedness[i][0].index

            cx = result.hand_landmarks[i][9].x - result.hand_world_landmarks[i][9].x
            cy = result.hand_landmarks[i][9].y - result.hand_world_landmarks[i][9].y
            cz = result.hand_landmarks[i][9].z*5 - result.hand_world_landmarks[i][9].z

            # Iterate handpoints
            for j in range(21):
                x = result.hand_world_landmarks[i][j].x + cx
                y = result.hand_world_landmarks[i][j].y + cy
                z = result.hand_world_landmarks[i][j].z + cz
                data[index].append(x)
                data[index].append(y)
                data[index].append(z)
        res = "&".join(map(str, data))
        socket.sendto( str.encode(str(res)) , serverAddressPort1 )

    # if result.gestures:
    #     gestos = ['None','None']
    #     for i in range(len(result.gestures)):
    #         index = 1 - result.handedness[i][0].index
    #         gestos[index] = result.gestures[i][0].category_name
    #     socket.sendto( str.encode(str(gestos)) , serverAddressPort2 )

# Parameters for Gesture Recognition
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='tasks/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2)


timestamp = 0
# Start Gesture Recognizer
with HandLandmarker.create_from_options(options) as recognizer:
    while video.isOpened(): 
        # Capture frames
        ret, frame = video.read()
        cv2.imshow('Camera', frame)
        if not ret:
            print("Ignoring empty frame")
            break
        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process frame
        recognizer.detect_async(mp_image, timestamp)

        if cv2.waitKey(5) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()