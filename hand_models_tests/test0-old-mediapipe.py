
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import socket

# Get (x,y) coordinates of a handpoint
def getXY(landmark, depth_image_flipped):
    x = int(landmark.x*len(depth_image_flipped[0]))
    y = int(landmark.y*len(depth_image_flipped))
    if x >= len(depth_image_flipped[0]):
        x = len(depth_image_flipped[0]) - 1
    if y >= len(depth_image_flipped):
        y = len(depth_image_flipped) - 1
    return (x, y)

# Visual configuration
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,50,255)
thickness = 1

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

# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    #static_image_mode = True,
    max_num_hands = 2,
    #min_detection_confidence = 0.5
)
mpDraw = mp.solutions.drawing_utils

# ====== Task ======
with open('tasks/model.task', 'rb') as f:
    model = f.read()
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ====== Socket ======
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort1 = ("127.0.0.1", 5052)
serverAddressPort2 = ("127.0.0.1", 5053)

# ====== Gesture Recognizer ======
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gestos = ['None','None']
    for i in range(len(result.gestures)):
        gestos[1-result.handedness[i][0].index] = result.gestures[i][0].category_name
        print(result.handedness[i][0].display_name, ":", result.gestures[i][0].category_name, end="   \t")
    print()
    socket.sendto( str.encode(str(gestos)) , serverAddressPort2 )
    return

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2) 

recognizer = GestureRecognizer.create_from_options(options)

# ====== Enable Streams ======
config.enable_device(device)

stream_res_x = 848 # 640 o 1280
stream_res_y = 480 # 480 o 720

stream_fps = 60 # 60 o 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device}")

while True:
    start_time = dt.datetime.today().timestamp()

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

    # Process hands
    
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
        data = [[],[]]
        number_of_hands = len(results.multi_hand_landmarks)
        i=0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            
            org2 = (20, org[1]+(20*(i+1)))
            hand_side_classification_list = results.multi_handedness[i]

            hand_side = hand_side_classification_list.classification[0].label
            index = hand_side_classification_list.classification[0].index

            middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]

            (x, y) = getXY(middle_finger_knuckle, depth_image_flipped)

            mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
            
            images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance:0.4} m away", org2, font, fontScale, color, thickness, cv2.LINE_AA)

            # Obtener coordenadas de los puntos de la mano
            for j in range(21):
                (x,y) = getXY(results.multi_hand_landmarks[i].landmark[j], depth_image_flipped)
                z = int(depth_image_flipped[y,x] * depth_scale *1000)
                y = stream_res_y - y
                data[index].append(x)
                data[index].append(y)
                data[index].append(z)

            i+=1
        
        # Send handpoints to socket
        delim = "&"
        res = delim.join(map(str, data))
        socket.sendto( str.encode(str(res)) , serverAddressPort1 )

        images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Recognize gesture

        color_images_rgb = cv2.flip(color_images_rgb,1) 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_images_rgb)
        recognizer.recognize_async(mp_image, int(round(time.time() * 1000)))


    else:
        images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)


    
        
    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff) if time_diff != 0 else 0
    org3 = (20, org[1] + 100)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)

    name_of_window = 'SN: ' + str(device)

    # Display images 
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break

print(f"Application Closing")
recognizer.close() # TASK
pipeline.stop()
print(f"Application Closed.")

