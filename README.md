# Text Detection Human Helper


## Parts Used
- LCD Display with HDMI
- Raspberry Pi 4
- Camera Module 3

## Python Packages
Python packages are all installed within the venv. These can be installed with the requirements.txt.

In addition, picamera2 must be installed via apt install since the package is directly connected to the camera module itself
`apt install python3-libcamera python3-picamera2 libcamera-apps`

These packages then need to be linked within the venv with the following command
`ln -s /usr/lib/python3/dist-packages/picamera2 venv/lib/python3.11/site-packages/`
`ln -s /usr/lib/python3/dist-packages/libcamera venv/lib/python3.11/site-packages/`
`ln -s /usr/lib/python3/dist-packages/pycamera venv/lib/python3.11/site-packages/ 2>/dev/null`

## To Run
1. Activate venv
2. Install packages from the requirements txt
`pip install -r requirements.txt`
3. Install the picamera2 module with `apt` and create symlinks.
4. Run ocr_main.py with `python3 ocr_main.py`

## Disclaimer
We used AI to debug our code and for finding Python packages suitable for our project.
