"""
MIT License

Copyright (c) 2023 Ye Yint Hmine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
import threading
import cv2
import json
import concurrent.futures
import math
import time
import winsound
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
from playsound import playsound

x_val = [0]
y_val = [0]

def load_classifications(file_path):
    """
    :param file_path: The file path of the classification data names json file
    :return: All the data class names for a specific data model
    """
    with open(file_path) as data_file:
        return json.load(data_file)["Classifications_YOLOv8n"]

def calculate_center(x1, y1, x2, y2):
    """
    :param x1, y1: the top left corner of a detected box
    :param x2, y2: the bottom right corner of a detected box
    :return: calculate the center point of the detected box
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return int(center_x), int(center_y)

def calculate_distance(x1, y1, x2, y2):
    """
    :param x1, y1: the first point of coordinate
    :param x2, y2: the second point of coordinate
    :return: calculate the distance between the first point and the second point
    """
    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    return distance

def reduce_cooldown(cooldown):
    """
    :param cooldown: the cooldown value to be operated on
    :return: the cooldown value after the operation
    """
    if cooldown > 0:
        cooldown -= 1
    return cooldown

def alert():
    """
    :return: Alert the user for distraction
    """
    frequency = 500
    duration = 500
    winsound.Beep(frequency, duration)


def updatePlot(x_val, y_val, x=1, y=0):
    x_val.append(x_val[-1] + x)
    y_val.append(y)
    if len(x_val) > 5:
        del x_val[0]
        del y_val[0]


def livePlot(i):
    global x_val, y_val


    plt.cla()

    plt.plot(x_val, y_val, label="Distraction")
    # plt.plot(x_val, high_y)
    plt.legend(loc="upper left")
    plt.ylim(-0.05, 1.05)

def startPlot():
    # set plt style
    plt.style.use("dark_background")
    plt.title("Distraction Graph")
    try:
        live_plt = FuncAnimation(plt.gcf(), livePlot, interval=100 * 1)
    except Exception:
        pass
    plt.show()

def main():
    """Detect distractions from YOLO and hand landmarks in a webcam stream."""
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    # Load YOLO model
    yolo_model = YOLO("../YOLO_models/yolov8n.pt")  # Replace with the path to the latest YOLO model
    yolo_model.predict(verbose=False)

    # Video stream setup
    video_capture = cv2.VideoCapture(1)
    #raw_capture = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 640, 360
    video_capture.set(3, WIDTH)
    video_capture.set(4, HEIGHT)
    #raw_capture.set(3, WIDTH)
    #raw_capture.set(4, HEIGHT)

    # Load classification data
    classification_data = load_classifications("utils/classification_names.json")

    # set the variables
    is_distracted = False
    cooldown_counter = 0
    cooldown_timer = 5

    frame_count = 0
    distraction_type = "phone"
    pTime = 0

    playsound('sounds/startup.wav')

    start_plot = threading.Thread(target=startPlot)
    start_plot.start()

    while True:

        # capture the video
        ret, frame = video_capture.read()
        ret_raw, raw_footage = video_capture.read()

        # mirror the frame for perfect coordination
        frame = cv2.flip(frame, 1)
        raw_footage = cv2.flip(raw_footage, 1)
        frame = cv2.convertScaleAbs(frame, alpha=.7, beta=14)
        raw_footage = cv2.convertScaleAbs(raw_footage, alpha=2, beta=6)

        if frame_count % 2 == 0:  # Process every 2nd frame
            # Run YOLO inference in a separate thread using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                yolo_future = executor.submit(yolo_model, frame.copy(), stream=True)

            # Process hand landmarks detection in the main thread
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(frame_rgb)

            # Get YOLO results
            yolo_results = yolo_future.result()

            index_finger = False

            # see if hand has been detected and calculate the detected position
            if hand_results.multi_hand_landmarks:
                index_finger = (
                    int(hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH),
                    int(hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
                )

            # run the result for the YOLO model
            for yolo_result in yolo_results:
                # Get detections
                detections = yolo_result.boxes
                for detection in detections:
                    classification = classification_data[int(detection.cls[0])]
                    if distraction_type == classification:
                        x1, y1, x2, y2 = detection.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        confidence = int(detection.conf[0] * 100)
                        center_x, center_y = calculate_center(x1, y1, x2, y2)

                        # Draw detection box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

                        cv2.putText(frame, f"{classification} {confidence}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), 3)
                        if index_finger:
                            distance = calculate_distance(center_x, center_y, index_finger[0], index_finger[1])
                            if confidence >=55:
                                if distance <= 150:
                                    is_distracted = True
                                    cooldown_counter = cooldown_timer




            if hand_results.multi_hand_landmarks:
                _ = [mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) for hand_landmarks in
                     hand_results.multi_hand_landmarks]

            updatePlot(x_val, y_val, x=1, y=int(is_distracted))

        if is_distracted:
            threading.Thread(target=alert).start()

        cooldown_counter = reduce_cooldown(cooldown_counter)
        if cooldown_counter == 0:
            is_distracted = False

        frame_count += 1

        current_time = time.time()
        fps = (1 / (current_time - pTime))
        pTime = current_time

        if is_distracted:
            cv2.putText(frame, f'Distracted: {str(is_distracted)}', (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Distracted: {str(is_distracted)}', (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0),
                        2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS: {str(int(fps))}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 0), 2, cv2.LINE_AA)

        final_frame = cv2.hconcat([frame, raw_footage])
        # cv2.imshow("ADWS footage", frame) to see just the ADWS footage
        # cv2.imshow("raw_footage", raw_footage) to see just the raw footage
        cv2.imshow("Whole frame", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    from Prereqs.settings import setting
    # INFO
    print(f"Autonomous Distraction Warning System - v{setting['version']}")
    print(f"For more information and releases, visit {setting['github']}. [This program uses MIT License]")

    main()
    sys.exit()
