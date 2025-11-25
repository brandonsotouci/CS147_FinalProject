import cv2
import pytesseract
from picamera2 import Picamera2
from pytesseract import Output
from time import sleep

picam2 = Picamera2()
picam2.preview_configuration.main.size=(980,540) # Configure window size
picam2.preview_configuration.main.format="RGB888" #8 bits
picam2.start()

#cap = cv2.VideoCapture(0)
#sleep(5)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


while True:
    # Capture frame-by-frame
    frame  = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #denoised = cv2.fastNlMeansDenoising(threshed, h=30)

    d = pytesseract.image_to_data(threshed, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # don't show empty text
            if text and text.strip() != "":
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
 
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
picam2.stop()
cv2.destroyAllWindows()
