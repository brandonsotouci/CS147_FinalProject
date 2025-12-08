from picamera2 import Picamera2
from pytesseract import Output
from time import sleep
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import signal
import time
import sys
import cv2 as cv
import numpy as np
import threading
import mediapipe as mp
import pytesseract
from datetime import datetime
import os 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
last_coordinates = None

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
frameToShow = None
selectedSample = None
currentPlaybackThread = None
processingFrame = False

row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 0)  # black
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Label box parameters
label_text_color = (255, 255, 255)  # white 
label_font_size = 1
label_thickness = 2

current_result = None

def process_result(result: GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
    global current_result
    current_result = result

picam2 = Picamera2()
picam2.preview_configuration.main.size=(980,540) # Configure window size
picam2.preview_configuration.main.format="RGB888" #8 bits
picam2.start()


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=process_result)      

with GestureRecognizer.create_from_options(options) as recognizer:
    last_display_time = time.time()
    target_fps = 30
    frame_interval = 1.0 / target_fps

    while True:
        frame = picam2.capture_array()
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        frame_ms = time.time_ns() // 1_000_000

        recognizer.recognize_async(mp_image, frame_ms)
        current_time = time.time()
        
        if current_result and (current_time - last_display_time) >= frame_interval:
            current_frame = frame.copy()
            frame_height, frame_width = current_frame.shape[:2]
        
            if current_result.gestures:
                for hand_index, gesture_list in enumerate(current_result.gestures):
                    
                    if gesture_list and hand_index < len(current_result.hand_landmarks):
                        gesture = gesture_list[0]
                        score = gesture.score

                        if score >= 0.7:

                            hand_landmarks = current_result.hand_landmarks[hand_index]
                            index_tip = hand_landmarks[8]
                            index_x_coordinate = int(index_tip.x * frame_width)
                            index_y_coordinate = int(index_tip.y * frame_height)

                            if last_coordinates is not None:
                                index_x_coordinate = int(last_coordinates[0] * 0.6 + index_x_coordinate * (1-0.6))
                                index_y_coordinate = int(last_coordinates[1] * 0.6 + index_y_coordinate * (1-0.6))

                            last_coordinates = (index_x_coordinate, index_y_coordinate)

                            x_min = max(0, index_x_coordinate)
                            y_min = max(0, index_y_coordinate)

                            x_max = min(frame_width, index_x_coordinate + 200)
                            y_max = min(frame_height, index_y_coordinate + 200)
                            frame_to_process = current_frame[y_min:y_max, x_min:x_max]

                            if frame_to_process.size > 0:
                                grayscaled = cv.cvtColor(frame_to_process, cv.COLOR_RGB2GRAY)

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                small_filepath = os.path.join("/home/humanic/CS147", f"roi_{timestamp}_200x200_AI.png")
                                big_filepath = os.path.join("/home/humanic/CS147", f"roi_{timestamp}_full_AI.png")

                                cv.imwrite(small_filepath, frame_to_process)
                                cv.imwrite(big_filepath, current_frame)

                                contrast = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                enhanced_box = contrast.apply(grayscaled)
                                _, threshold_frame = cv.threshold(enhanced_box, 150, 255, cv.THRESH_BINARY)
                                text_processed = pytesseract.image_to_string(threshold_frame, config='--psm 4')
                                text = text_processed.strip()
                                if text:
                                    print("TEXT FOUND: ", text)


            cv.imshow('gesture_recognition', current_frame)
            last_display_time = current_time

        if cv.waitKey(1) == ord('q'):
            break



cv.destroyAllWindows()

# When everything done, release the capture
picam2.stop()

