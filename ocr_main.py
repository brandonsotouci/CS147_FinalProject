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
# nltk.download('omw-1.4')  # Open Multilingual WordNet
# nltk.download('wordnet')

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

async def translate(word, definition):
    async with Translator() as translator:
        word_translated = await translator.translate(word, src=LANGUAGE_TO_TRANSLATE, dest=TRANSLATE_TO_LANGUAGE)
        definition_translated = await translator.translate(definition, src=LANGUAGE_TO_TRANSLATE, dest=TRANSLATE_TO_LANGUAGE)
        print(f'{word_translated.text}: {definition_translated.text}')


def define_word(word):
    synsets = wordnet.synsets(word)
    if synsets:
        word_definition = synsets[0].definition()  # First definition
        print(f'{word}: {word_definition}')
        asyncio.run(translate(word, word_definition))
    else:
        print("Word not in dictionary. Seek other library")

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
EnchantDictEnglish = enchant.Dict("en_US")
PyDict = PyDictionary()

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

                        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        # small_filepath = os.path.join("/home/humanic/CS147", f"roi_{timestamp}_200x200_AI.png")
                        # big_filepath = os.path.join("/home/humanic/CS147", f"roi_{timestamp}_full_AI.png")

                        # cv.imwrite(small_filepath, frame_to_process)
                        # cv.imwrite(big_filepath, current_frame)

                        contrast = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_box = contrast.apply(grayscaled)
                        _, threshold_frame = cv.threshold(enhanced_box, 150, 255, cv.THRESH_BINARY)
                        results = reader.readtext(frame_to_process)
                        text_processed = ""

                        # The output will be in a list format, each item represents a bounding box, the text detected and confident level, respectively.
                        # Ex. [([[189, 75], [469, 75], [469, 165], [189, 165]], 'Omomo', 0.3754989504814148)]

                        for detection in results:
                            text = detection[1]
                            text_processed += text + "\n"

                        if text_processed:
                            print("TEXT FOUND: ", text)
                            word = clean_text(text)

                            isValidWord = EnchantDictEnglish.check(word)

                            if isValidWord:
                                define_word(word)
                            else:
                                print(f"{word} IS NOT A VALID WORD")
                        else:
                            print("NO TEXT FOUND")


            cv.imshow('gesture_recognition', current_frame)
            last_display_time = current_time

        if cv.waitKey(1) == ord('q'):
            break



cv.destroyAllWindows()

# When everything done, release the capture
picam2.stop()

