from picamera2 import Picamera2
from libcamera import controls
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
import easyocr
from datetime import datetime
import os 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
last_coordinates = None

def signal_handler(sig, frame):
    sys.exit(0)

def visualize_finger(current_frame, index_x_coordinate, index_y_coordinate):
    cv.circle(current_frame, (index_x_coordinate, index_y_coordinate), 15, (0, 0, 255), -1)  # Red filled circle
    cv.circle(current_frame, (index_x_coordinate, index_y_coordinate), 20, (255, 255, 255), 2)  # White outline
    # cv.putText(current_frame, "INDEX", (index_x_coordinate + 25, index_y_coordinate), 
    #     cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def is_index_extended(hand_landmarks):
    index_tip = hand_landmarks[8]
    index_mcp = hand_landmarks[5]

    return index_tip.y < index_mcp.y


def is_pointing(hand_landmarks):
    """
    Detect if hand is pointing from any angle.
    Checks if index finger is extended and roughly in line with hand direction.

    Args:
        hand_landmarks: List of 21 hand landmarks from MediaPipe

    Returns:
        bool: True if hand is pointing, False otherwise
    """
    # Landmark indices
    WRIST = 0
    INDEX_MCP = 5      # Index finger middle joint
    INDEX_PIP = 6      # Index finger proximal joint
    INDEX_TIP = 8      # Index finger tip
    MIDDLE_MCP = 9     # Middle finger middle joint
    MIDDLE_PIP = 10    # Middle finger proximal joint
    MIDDLE_TIP = 12    # Middle finger tip

    # Get coordinates
    wrist = hand_landmarks[WRIST]
    index_mcp = hand_landmarks[INDEX_MCP]
    index_pip = hand_landmarks[INDEX_PIP]
    index_tip = hand_landmarks[INDEX_TIP]
    middle_pip = hand_landmarks[MIDDLE_PIP]
    middle_tip = hand_landmarks[MIDDLE_TIP]

    # Calculate distances
    index_length = np.sqrt((index_tip.x - index_mcp.x)**2 +
                          (index_tip.y - index_mcp.y)**2 +
                          (index_tip.z - index_mcp.z)**2)

    middle_length = np.sqrt((middle_tip.x - middle_pip.x)**2 +
                           (middle_tip.y - middle_pip.y)**2 +
                           (middle_tip.z - middle_pip.z)**2)

    # Index finger should be extended (longer than middle finger)
    if index_length < middle_length * 0.8:
        return False

    # Calculate vector from wrist to index tip
    wrist_to_index = np.array([index_tip.x - wrist.x,
                               index_tip.y - wrist.y,
                               index_tip.z - wrist.z])

    # Calculate vector from wrist to middle tip
    wrist_to_middle = np.array([middle_tip.x - wrist.x,
                                middle_tip.y - wrist.y,
                                middle_tip.z - wrist.z])

    # Calculate angle between vectors using dot product
    dot_product = np.dot(wrist_to_index, wrist_to_middle)
    mag1 = np.linalg.norm(wrist_to_index)
    mag2 = np.linalg.norm(wrist_to_middle)

    if mag1 == 0 or mag2 == 0:
        return False

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)

    # Index finger should be significantly separated from middle finger (> 30 degrees)
    return angle_degrees > 30


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

#picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    # running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    # result_callback=process_result
)      

reader = easyocr.Reader(['en'])

with GestureRecognizer.create_from_options(options) as recognizer:
    last_display_time = time.time()
    target_fps = 30
    frame_interval = 1.0 / target_fps

    last_capture_time = 0
    capture_interval = 5

    while True:
        frame = picam2.capture_array()
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        frame_ms = time.time_ns() // 1_000_000

        # recognizer.recognize_async(mp_image, frame_ms)
        current_time = time.time()
        current_result = recognizer.recognize_for_video(mp_image, frame_ms)
        
        if current_result:  # Ask brandon about frame_interval??
            current_frame = frame.copy()
            frame_height, frame_width = current_frame.shape[:2]
        
            if current_result.hand_landmarks:
                hand_landmarks = current_result.hand_landmarks[0]
                index_tip = hand_landmarks[8]
                index_x_coordinate = int(index_tip.x * frame_width)
                index_y_coordinate = int(index_tip.y * frame_height)

                visualize_finger(current_frame, index_x_coordinate, index_y_coordinate)
                
                # if (is_pointing(hand_landmarks)): 
                if (is_index_extended(hand_landmarks) and current_time - last_capture_time >= capture_interval):
                    last_capture_time = current_time

                    print("Pointing detected and 5 second interval")

                    box_width = 200
                    box_height = 200

                    # Box positioned above the fingertip
                    offset_above = 0  # How many pixels above the finger

                    x_min = max(0, index_x_coordinate - box_width // 2)
                    y_min = max(0, index_y_coordinate - box_height - offset_above)  # Above the finger

                    x_max = min(frame_width, index_x_coordinate + box_width // 2)
                    y_max = min(frame_height, index_y_coordinate - offset_above)

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
                        results = reader.readtext(frame_to_process)
                        text_processed = ""

                        for detection in results:
                            text = detection[1]
                            text_processed += text + "\n"


                        #text_processed = pytesseract.image_to_string(threshold_frame, config='--psm 6')

                        #text = text_processed.strip()
                        if text_processed:
                            print("TEXT FOUND: ", text)


            cv.imshow('gesture_recognition', current_frame)
            last_display_time = current_time

        if cv.waitKey(1) == ord('q'):
            break



cv.destroyAllWindows()

# When everything done, release the capture
picam2.stop()

