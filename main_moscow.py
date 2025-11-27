from XInput import *
from ultralytics import YOLO
import time
import serial.tools.list_ports
from read_controller import XboxController
import socket
ports = serial.tools.list_ports.comports()
qr_mode = False
line_mode = False
import cv2
import threading
video=cv2.VideoCapture(1)
qcd = cv2.QRCodeDetector()
video.set(3, 1920)
video.set(4, 1080)
port = "COM4"
baudrate = 57600
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Ширина: {width}, Высота: {height}")
controller = XboxController()
line_mode=False
import numpy as np
index = 0
l_stick = (127, 127)
r_stick = (127, 127)
left=127
right=127

model = YOLO('tt.pt')


def calculate_crc(data):
    crc = 0
    for byte in data:
        crc ^= byte  # Применяем XOR для каждого байта
    if crc == 255:
        crc = 0
    return crc



UDP_IP ='192.168.10.94' #"192.168.4.1"  # IP-адрес ESP32 в режиме точки доступа
UDP_PORT = 12345        # Порт, на который отправляем данные

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


err_regulated=127

def rescale_frame(frame, percent=75):
    #width = int(frame.shape[1] * percent/ 100)
    #height = int(frame.shape[0] * percent/ 100)
    #dim = (1540, 800)
    dim = (1540, 795)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_LINEAR)

while True:

    controller.read_controller()

    xbox_data = controller.get_values()

    button_states = [xbox_data.a_button, xbox_data.b_button, xbox_data.x_button, xbox_data.y_button,
                     xbox_data.right_button, xbox_data.left_button, xbox_data.dpad_up, xbox_data.dpad_right,
                     xbox_data.dpad_left, xbox_data.dpad_down,
                     xbox_data.qr_mode,
                     xbox_data.line_mode]
    byte1 = 0
    byte2 = 0
    for i in range(12):
        # НИЩАДНО пакуем значения кнопок в 2 байта #
        if button_states[i]:  # Если значение True
            if i < 7:  # Первые 7 кнопок в byte1
                byte1 |= (1 << i)
            else:  # Остальные 5 кнопок в byte2
                byte2 |= (1 << (i - 7))
    #print(byte1,byte2)
    packed_shoulder = (xbox_data.l_trigger & 0x0F) | ((xbox_data.r_trigger & 0x0F) << 4)

    if line_mode:
        otpr = [xbox_data.r_stick[0], right, xbox_data.l_stick[0], left, byte1, byte2,
                packed_shoulder]
    else:
        otpr = [xbox_data.r_stick[0], xbox_data.r_stick[1], xbox_data.l_stick[0], xbox_data.l_stick[1], byte1, byte2,
                packed_shoulder]
    crc_value = calculate_crc(otpr)

    # Формируем пакет для отправки
    str_otpr = [255] + otpr + [crc_value]
    print(str_otpr)
    line_mode  = xbox_data.line_mode
    qr_mode = xbox_data.qr_mode
    sign_mode= xbox_data.a_button
    # Отправка данных через UDP
    sock.sendto(bytearray(str_otpr), (UDP_IP, UDP_PORT))

    hasFrame, frame = video.read()

    if qr_mode:

        h, w = frame.shape[:2]
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        #frame = cv2.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)

        #x, y, w, h = roi
        #frame = frame[y:y + h - 50, x + 70:x + w - 20]

        cv2.putText(frame, "QR_MODE", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ((0, 255, 0)), 2,
                    cv2.LINE_AA)
        if (qr_text!=""):
            cv2.putText(frame, qr_text[0], (100, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1,
                        cv2.LINE_AA)
        retval, decoded_info, points, straigt_qr_code = qcd.detectAndDecodeMulti(frame)
        if retval:
            cv2.polylines(frame, points.astype(int), True, (0, 255, 0),3)

            if ((decoded_info[0]!='')and(qr_text=="")):
                qr_text=decoded_info


    else:
        qr_text=""
    if sign_mode:
        cv2.putText(frame, "SIGN_MODE", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ((25, 255, 0)), 2,
                    cv2.LINE_AA)
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                # Рисуем прямоугольник и подпись
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Отображаем кадр с результатами
    if line_mode:
        # 1) draw mode label
        cv2.putText(frame, "LINE_MODE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # 2) crop to bottom third of the frame
        h, w = frame.shape[:2]
        y0 = int(h * 2 / 3)
        roi = frame[y0:h, :]

        # 3) grayscale + blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4) automatic threshold (Otsu’s method)
        _, mask = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 5) clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 6) find contours & pick the largest
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                # centroid of the contour
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # draw it (offset by y0)
                cv2.circle(frame, (cx, cy + y0), 5, (255, 255, 255), -1)
                cv2.drawContours(roi, [c], -1, (0, 255, 0), 1)

                # 7) simple proportional control
                error = (w // 2 - cx)
                kp = 0.1
                right = int(np.clip(127 + kp * error + 28, 0, 254))
                left = int(np.clip(127 - kp * error + 28, 0, 254))

        else:
            print("I don't see the line")

    frame150 = rescale_frame(frame, percent=150)
    try:
        cv2.imshow("Face detection", frame150)
        # если кадра нет
        if not hasFrame:
            # останавливаемся и выходим из цикла
            cv2.waitKey()
    except:
        video.release()
    if cv2.waitKey(1) & 0xFF == ord('l'):
        cv2.destroyAllWindows()

    time.sleep(0.02)
