import mediapipe as mp
import pyrealsense2 as rs
import cv2
import datetime as dt
import socket
import numpy as np
import time


profundidad = []
escala = 0
def getXY(landmark, depth_image_flipped):
    x = int(landmark.x*len(depth_image_flipped[0]))
    y = int(landmark.y*len(depth_image_flipped))
    if x >= len(depth_image_flipped[0]):
        x = len(depth_image_flipped[0]) - 1
    if y >= len(depth_image_flipped):
        y = len(depth_image_flipped) - 1
    return (x, y)


# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0]
pipeline = rs.pipeline()
config = rs.config()

# ====== Sockets ======
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort1 = ("127.0.0.1", 5052)
serverAddressPort2 = ("127.0.0.1", 5053)

# ====== Gesture Recognition ======
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ====== Gesture Recognition Results ======
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #print('gesture recognition result: {}'.format(result))

    # cv2.imshow("show", output_image.numpy_view())
    # key = cv2.waitKey(5) & 0xFF

    if result.hand_landmarks:
        data = [[],[]]
        # Iterate hands
        for i in range(len(result.hand_world_landmarks)):

            index = 1 - result.handedness[i][0].index

            (cx, cy) = getXY(result.hand_landmarks[i][0], profundidad) 
            cz = int(profundidad[cy,cx] * escala *1000) 
            rs_x = cx / 500
            rs_y = cy / 500
            rs_z = cz / 200

            mp_x = result.hand_landmarks[i][9].x - result.hand_landmarks[i][9].x
            mp_y = result.hand_landmarks[i][9].y - result.hand_landmarks[i][9].y
            mp_z = result.hand_landmarks[i][9].z - result.hand_landmarks[i][9].z

            # Iterate handpoints
            for j in range(21):
                x = result.hand_world_landmarks[i][j].x + mp_x
                y = result.hand_world_landmarks[i][j].y + mp_y
                z = result.hand_world_landmarks[i][j].z + rs_z
                data[index].append(x)
                data[index].append(y)
                data[index].append(z)
                print(x, y, z)

        res = "&".join(map(str, data))
        socket.sendto( str.encode(str(res)) , serverAddressPort1 )
        

    if result.gestures:
        gestos = ['None','None']
        for i in range(len(result.gestures)):
            index = 1 - result.handedness[i][0].index
            gestos[index] = result.gestures[i][0].category_name
        socket.sendto( str.encode(str(gestos)) , serverAddressPort2 )

# ====== Gesture Recognition ======
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='tasks/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_hand_detection_confidence = 0.01,
    min_hand_presence_confidence = 0.01,
    min_tracking_confidence = 0.01,
    num_hands=2)

# ====== Start Realsense ======
# Enable Streams
config.enable_device(device)
stream_res_x = 848 # 640 o 1280
stream_res_y = 480 # 480 o 720
stream_fps = 60 # 60 o 30
config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
# Get depth Scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# Set clipping distance
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Start Loop ======
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:

        # Get and align frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        # Process images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = cv2.flip(color_image,1) # background_removed (para mostrar imagen coloreada sin backgorund) , color_image (para mostrar imagen colorada con backgound)
        color_image = cv2.flip(color_image,1) 
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_images_rgb)

        # Process hand
        profundidad = depth_image_flipped
        escala = depth_scale
        recognizer.recognize_async(mp_image, int(round(time.time() * 1000)))

        # Display window
        cv2.imshow("Realsense " + str(device), images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {device}")
            break

recognizer.close()
pipeline.stop()
socket.close()
cv2.destroyAllWindows()