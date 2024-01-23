# Hand Gesture Recognition App
Aplicacion para reconocer manos utilizando un Intel Realsense D455 como camara y sensor de profundidad, y Mediapipe como modelo para detectar manos y gestos.
Detecta la estrucura de la mano, coordenadas de los puntos, su profundidad y el gesto reconocido.
![Screenshot](screenshots/image.png)

## Test 1: Mediapipe: GestureRecognizer

## Test 2: Mediapipe: HandLandmarker

## Test 3: Holist

gfdgfdgfd
``` python
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0]
pipeline = rs.pipeline()
```