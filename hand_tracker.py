import mediapipe as mp
import pyrealsense2 as rs
import cv2
import datetime as dt
import socket
import numpy as np
import time

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
previous_z = [0.4, 0.4] 

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


# ====== Gesture Recognition ======
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='tasks/gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence = 0.3,
    min_hand_presence_confidence = 0.1,
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
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        image = np.copy(color_image)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_images_rgb)

        # Process hand
        result = recognizer.recognize(mp_image)

        # Analize ressults
        if result.hand_landmarks:
            data = [[],[]]
            # Iterate hands
            for i in range(len(result.hand_world_landmarks)):

                index = 1 - result.handedness[i][0].index

                center_x = int(result.hand_landmarks[i][0].x * len(depth_image[0]))
                center_y = int(result.hand_landmarks[i][0].y * len(depth_image))
                if center_x >= len(depth_image[0]):
                    center_x = len(depth_image[0]) - 1
                if center_y >= len(depth_image):
                    center_y = len(depth_image) - 1
                

                # Filter wrong depth results
                center_z = depth_image[center_y,center_x] * depth_scale
                if center_z >= 0.3 and center_z <= 1.3:
                    previous_z[index] = center_z
                center_z = previous_z[index]
                
                # Adjust results
                center_x = center_x / 1000
                center_y = center_y / 1000
                center_z = center_z 

                # Iterate handpoints: hand_landmarks has better results for x and y. hand_wold_landmarks has better results for z
                for j in range(21):
                    x = result.hand_landmarks[i][j].x * stream_res_x / 900 + center_x
                    y = result.hand_landmarks[i][j].y * stream_res_y / 900 + center_y 
                    # x = result.hand_world_landmarks[i][j].x + center_x 
                    # y = result.hand_world_landmarks[i][j].y + center_y
                    z = result.hand_world_landmarks[i][j].z + center_z 
                    data[index].append(x)
                    data[index].append(y)
                    data[index].append(z)
                    cv2.circle(image, (int(result.hand_landmarks[i][j].x * stream_res_x), int(result.hand_landmarks[i][j].y*stream_res_y)), 2, (0,255,0), 2)
                    cv2.circle(image, (int(result.hand_world_landmarks[i][j].x* 1000 + center_x*1000), int(result.hand_world_landmarks[i][j].y*1000 + center_y*1000)), 2, (255,0,0), 2)
                    

            res = "&".join(map(str, data))
            socket.sendto( str.encode(str(res)) , serverAddressPort1 )
        

        if result.gestures:
            gestos = ['None','None']
            for i in range(len(result.gestures)):
                index = 1 - result.handedness[i][0].index
                gestos[index] = result.gestures[i][0].category_name
            socket.sendto( str.encode(str(gestos)) , serverAddressPort2 )


        # Display window
        cv2.imshow("RealsenseRGB " + str(device), color_image)
        cv2.imshow("RealsenseDepth " + str(device), depth_colormap)
        cv2.imshow("Mediapipe Results", image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {device}")
            break

recognizer.close()
pipeline.stop()
socket.close()
cv2.destroyAllWindows()