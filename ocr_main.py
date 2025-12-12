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
import enchant
from PyDictionary import PyDictionary
import easyocr
from datetime import datetime
import os 
import string

import asyncio
from googletrans import Translator
TRANSLATE_TO_LANGUAGE = 'es' # look at googletrans.languages
LANGUAGE_TO_TRANSLATE = 'en'

import nltk
from nltk.corpus import wordnet

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
last_coordinates = None

def signal_handler(sig, frame):
    sys.exit(0)

def visualize_finger(current_frame, index_x_coordinate, index_y_coordinate):
    cv.circle(current_frame, (index_x_coordinate, index_y_coordinate), 15, (0, 0, 255), -1)  # Red filled circle
    cv.circle(current_frame, (index_x_coordinate, index_y_coordinate), 20, (255, 255, 255), 2)  # White outline
   
async def translate(word, definition):
    async with Translator() as translator:
        word_translated = await translator.translate(word, src=LANGUAGE_TO_TRANSLATE, dest=TRANSLATE_TO_LANGUAGE)
        definition_translated = await translator.translate(definition, src=LANGUAGE_TO_TRANSLATE, dest=TRANSLATE_TO_LANGUAGE)
        return word_translated.text, definition_translated.text

def clean_text(word):
    return word.strip(string.punctuation)

def is_index_extended(hand_landmarks):
    index_tip = hand_landmarks[8]
    index_mcp = hand_landmarks[5]

    return index_tip.y < index_mcp.y

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

row_size = 50
left_margin = 24
text_color = (0, 0, 0)
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Label box parameters
label_text_color = (255, 255, 255)
label_font_size = 1
label_thickness = 2

current_result = None

def process_result(result: GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
    global current_result
    current_result = result

picam2 = Picamera2()
picam2.preview_configuration.main.size=(1200, 800) # window size
picam2.preview_configuration.main.format="RGB888"
picam2.start()

picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)      

reader = easyocr.Reader(['en'])
EnchantDictEnglish = enchant.Dict("en_US")
PyDict = PyDictionary()

def wrap_text(text, font, font_scale, thickness, max_width):
    words = text.split()
    lines = []
    line = ""
    
    for word in words:
        test_line = line + word + " "
        text_size = cv.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] > max_width:
            if line:
                lines.append(line.strip())
            line = word + " "
        else:
            line = test_line
    
    if line:
        lines.append(line.strip())
    
    return lines

def safe_putText(display, text, x, y, font, fontScale, color, thickness, max_width=None):
    h = display.shape[0]
    
    if max_width:
        lines = wrap_text(text, font, fontScale, thickness, max_width)
        for line in lines:
            if y < h:
                cv.putText(display, line, (x, y), font, fontScale, color, thickness)
                y += int(fontScale * 30)
    else:
        if y < h:
            cv.putText(display, text, (x, y), font, fontScale, color, thickness)

def create_display(frame, information):
    height, width = frame.shape[:2]
    
    camera_width = int(width * 2 / 3)
    box_width = width - camera_width
    left_side = cv.resize(frame, (camera_width, height))

    right_side = np.full((height, box_width, 3), 128, dtype=np.uint8)
    display = np.hstack([left_side, right_side])
    
    text_color = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX
    
    safe_putText(display, "Text Processed", camera_width + 20, 40, font, 1, text_color, 2)
    cv.line(display, (camera_width + 20, 60), (width - 20, 60), text_color, 2)
    
    if information:
        y_offset = 100
        max_width = box_width - 40
        safe_putText(display, "Word: " + information["Word"], camera_width + 20, y_offset, font, 1.0, text_color, 2)
        y_offset += 50
        
        if information.get("Definition", None) is not None:
            safe_putText(display, "Definition: " + information["Definition"], camera_width + 20, y_offset, font, 0.75, 
                        text_color, 2, max_width=max_width)
        y_offset += 150
        if information.get("Spanish Word", None) is not None:
            safe_putText(display, "Spanish Word: " + information["Spanish Word"], camera_width + 20, y_offset, font, 0.75, 
                        text_color, 2, max_width=max_width)

        y_offset += 150
        if information.get("Translated Definition", None) is not None:
            safe_putText(display, "Translated Definition: " + information["Translated Definition"], camera_width + 20, y_offset, font, 0.75, 
                        text_color, 2, max_width=max_width)
    
    return display

information = {}
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

        current_time = time.time()
        current_result = recognizer.recognize_for_video(mp_image, frame_ms)
        if current_result:  
            current_frame = frame.copy()
            frame_height, frame_width = current_frame.shape[:2]
        
            if current_result.hand_landmarks:
                hand_landmarks = current_result.hand_landmarks[0]
                index_tip = hand_landmarks[8]
                index_x_coordinate = int(index_tip.x * frame_width)
                index_y_coordinate = int(index_tip.y * frame_height)

                visualize_finger(current_frame, index_x_coordinate, index_y_coordinate)
                
                if (is_index_extended(hand_landmarks) and current_time - last_capture_time >= capture_interval):
                    last_capture_time = current_time

                    #print("Pointing detected and 5 second interval")

                    box_width = 200
                    box_height = 200
                    offset_above = 0
                    x_min = max(0, index_x_coordinate - box_width // 2)
                    y_min = max(0, index_y_coordinate - box_height - offset_above)

                    x_max = min(frame_width, index_x_coordinate + box_width // 2)
                    y_max = min(frame_height, index_y_coordinate - offset_above)

                    frame_to_process = current_frame[y_min:y_max, x_min:x_max]

                    if frame_to_process.size > 0:
                        grayscaled = cv.cvtColor(frame_to_process, cv.COLOR_RGB2GRAY)
                        contrast = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_box = contrast.apply(grayscaled)
                        _, threshold_frame = cv.threshold(enhanced_box, 150, 255, cv.THRESH_BINARY)
                        results = reader.readtext(frame_to_process)
                        text_processed = ""

                        # The output will be in a list format, each item represents a bounding box, the text detected, and confident level.
                        # Ex. [([[189, 75], [469, 75], [469, 165], [189, 165]], 'Omomo', 0.3754989504814148)]

                        clear_word, curr_score = "", 0
                        for detection in results:
                            curr_word, curr_weight = detection[1], detection[2]
                            if curr_weight > curr_score:
                                curr_score = curr_weight
                                clear_word = curr_word.strip()

                        if clear_word:
                            word = clean_text(clear_word)
                            #print("Last Text: ", word)
                            information = {}
                            information["Word"] = word
                            if word and EnchantDictEnglish.check(word):
                                synsets = wordnet.synsets(word)
                                if synsets:
                                    information["Definition"] = str(synsets[0].definition())
                                    spanish_word, spanish_definition = asyncio.run(translate(word, information["Definition"]))
                                    information["Spanish Word"] = str(spanish_word)
                                    information["Translated Definition"] = str(spanish_definition)

            cv.imshow('gesture_recognition', create_display(current_frame, information))
            last_display_time = current_time

        if cv.waitKey(1) == ord('q'):
            break

cv.destroyAllWindows()
picam2.stop()

