import re
import subprocess
import os
import numpy as np
import time
import requests
import threading
import bluetooth
import cv2
import smbus
from datetime import datetime
import base64
import traceback
import random
import speech_recognition as sr
import webrtcvad
import pyaudio
from picamera2 import Picamera2
import math
import glob

def clear_files_in_folder(folder_path):
    # Pattern to match all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))

    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted file: {file}")

# Example usage
folder_path = 'Pictures/'
clear_files_in_folder(folder_path)
move_set = []


# Initialize the recognizer and VAD with the highest aggressiveness setting
r = sr.Recognizer()
vad = webrtcvad.Vad(3)  # Highest sensitivity
print("Initialized recognizer and VAD.")
# Audio stream parameters
CHUNK = 320  # 20 ms of audio at 16000 Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(FORMAT)



p = pyaudio.PyAudio()
is_transcribing = False  # Global flag to control microphone input
the_list = []
print("Audio stream configured.")
file = open('playback_text.txt','w+')
file.write('')
file.close()
file = open('last_phrase.txt','w+')
file.write('')
file.close()
file = open('last_phrase2.txt','w+')
file.write('')
file.close()
file = open('last_phrase3.txt','w+')
file.write('')
file.close()
# Dictionary of known object heights (in meters)
known_object_heights = {
    'person': 1.7,
    'bicycle': 1.0,
    'car': 1.5,
    'motorcycle': 1.1,
    'airplane': 10.0,      # Average for small aircraft
    'bus': 3.0,
    'train': 4.0,           # Per car
    'truck': 2.5,
    'boat': 2.0,
    'traffic light': 0.6,
    'fire hydrant': 0.5,
    'stop sign': 0.75,
    'parking meter': 1.2,
    'bench': 1.5,
    'bird': 0.3,
    'cat': 0.25,
    'dog': 0.6,
    'horse': 1.6,
    'sheep': 0.8,
    'cow': 1.5,
    'elephant': 3.0,
    'bear': 1.2,
    'zebra': 1.4,
    'giraffe': 5.5,
    'backpack': 0.5,
    'umbrella': 1.0,
    'handbag': 0.3,
    'tie': 0.5,
    'suitcase': 0.7,
    'frisbee': 0.3,
    'skis': 1.8,
    'snowboard': 1.6,
    'sports ball': 0.22,    # e.g., soccer ball
    'kite': 1.2,
    'baseball bat': 1.1,
    'baseball glove': 0.35,
    'skateboard': 0.8,
    'surfboard': 2.0,
    'tennis racket': 1.0,
    'bottle': 0.3,
    'wine glass': 0.3,
    'cup': 0.15,
    'fork': 0.2,
    'knife': 0.3,
    'spoon': 0.2,
    'bowl': 0.2,
    'banana': 0.2,
    'apple': 0.1,
    'sandwich': 0.2,
    'orange': 0.1,
    'broccoli': 0.3,
    'carrot': 0.3,
    'hot dog': 0.2,
    'pizza': 0.3,
    'donut': 0.15,
    'cake': 0.3,
    'chair': 0.9,
    'sofa': 2.0,
    'potted plant': 0.5,
    'bed': 2.0,
    'dining table': 1.8,
    'toilet': 0.6,
    'tv monitor': 1.2,
    'laptop': 0.4,
    'mouse': 0.15,
    'remote': 0.15,
    'keyboard': 0.5,
    'cell phone': 0.15,
    'microwave': 0.6,
    'oven': 0.8,
    'toaster': 0.3,
    'sink': 0.5,
    'refrigerator': 1.8,
    'book': 0.25,
    'clock': 0.3,
    'vase': 0.4,
    'scissors': 0.3,
    'teddy bear': 0.3,
    'hair dryer': 0.3,
    'toothbrush': 0.2
}

# Default height for unknown classes (in meters)
default_height = 1.0

# -----------------------------------
# 2. Camera Calibration Parameters
# -----------------------------------

# Camera parameters (these should be obtained from calibration)
focal_length_px = 2050 

# -----------------------------------
# 3. Distance Estimation Function
# -----------------------------------

def estimate_distance(focal_length, real_height, pixel_height):
    """
    Estimate the distance from the camera to the object using the pinhole camera model.
    
    Parameters:
    - focal_length (float): Focal length of the camera in pixels.
    - real_height (float): Real-world height of the object in meters.
    - pixel_height (float): Height of the object's bounding box in the image in pixels.
    
    Returns:
    - float: Estimated distance in meters.
    """
    if pixel_height == 0:
        return float('inf')  # Avoid division by zero
    return ((focal_length * real_height) / pixel_height)/6

# -----------------------------------
# 4. Position and Size Description Functions
# -----------------------------------

def get_position_description(x, y, width, height):
    """
    Return a text description of the position based on coordinates in a 5x5 grid.
    
    Parameters:
    - x (float): X-coordinate of the object's center in the image.
    - y (float): Y-coordinate of the object's center in the image.
    - width (float): Width of the image.
    - height (float): Height of the image.
    
    Returns:
    - str: Description of the object's position.
    """
    # Determine horizontal position in 5 sections
    if x < width / 5:
        horizontal = "Turn Left 45 Degrees"
    elif x < 2 * width / 5:
        horizontal = "Turn Left 15 Degrees"
    elif x < 3 * width / 5:
        horizontal = "already centered between left and right"
    elif x < 4 * width / 5:
        horizontal = "Turn Right 15 Degrees"
    else:
        horizontal = "Turn Right 45 Degrees"
    
    # Determine vertical position in 5 sections
    if y < height / 5:
        vertical = "Raise Camera Angle"
    elif y < 2 * height / 5:
        vertical = "Raise Camera Angle"
    elif y < 3 * height / 5:
        vertical = "already centered on the vertical"
    elif y < 4 * height / 5:
        vertical = "Lower Camera Angle"
    else:
        vertical = "Lower Camera Angle"
    
    # Combine the horizontal and vertical descriptions
    if horizontal == "already centered between left and right" and vertical == "already centered on the vertical":
        return "already centered on object"
    else:
        return f"{vertical} and {horizontal}"



with open("last_phrase2.txt","w+") as f:
    f.write('')
# -----------------------------------
# 5. Overlap Removal Function
# -----------------------------------

def remove_overlapping_boxes(boxes, class_ids, confidences):
    """
    Remove overlapping boxes of the same class, keeping only the one with the highest confidence.
    
    Parameters:
    - boxes (list): List of bounding boxes [x, y, w, h].
    - class_ids (list): List of class IDs corresponding to each box.
    - confidences (list): List of confidence scores corresponding to each box.
    
    Returns:
    - tuple: Filtered lists of boxes, class_ids, and confidences.
    """
    final_boxes = []
    final_class_ids = []
    final_confidences = []

    for i in range(len(boxes)):
        keep = True
        for j in range(len(final_boxes)):
            if class_ids[i] == final_class_ids[j]:
                box1 = boxes[i]
                box2 = final_boxes[j]

                x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
                x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

                # Calculate the overlap area
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)

                # Calculate overlap ratio
                overlap_ratio = inter_area / min(box1_area, box2_area)

                if overlap_ratio > 0.5:
                    if confidences[i] > final_confidences[j]:
                        final_boxes[j] = box1
                        final_confidences[j] = confidences[i]
                    keep = False
                    break

        if keep:
            final_boxes.append(boxes[i])
            final_class_ids.append(class_ids[i])
            final_confidences.append(confidences[i])

    return final_boxes, final_class_ids, final_confidences

# -----------------------------------
# 6. YOLO Detection Function with Distance Estimation
# -----------------------------------



def yolo_detect():
    global chat_history
    """
    Perform YOLO object detection on an image, estimate distances to detected objects,
    and update chat history with descriptive information.
    
    Returns:
    - None
    """
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    
    # Dictionary to store times
    time_logs = {}
    
    # Start time tracking
    total_start = time.time()

    try:
        # Load image
        start = time.time()
        img = cv2.imread('this_temp.jpg')
        if img is None:
            print("Error: Image 'this_temp.jpg' not found.")
            return
        height, width, channels = img.shape
        time_logs['Load Image'] = time.time() - start

        # Prepare the image for YOLO
        start = time.time()
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        time_logs['YOLO Forward'] = time.time() - start

        # Extract bounding boxes and confidences
        start = time.time()
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.35:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        time_logs['Extract Bounding Boxes'] = time.time() - start

        # Remove overlapping boxes
        start = time.time()
        boxes, class_ids, confidences = remove_overlapping_boxes(boxes, class_ids, confidences)
        time_logs['Remove Overlapping Boxes'] = time.time() - start

        # Annotate image and generate descriptions
        start = time.time()
        descriptions = []

        # Define center grid rectangle
        center_x_min = 2 * width / 5
        center_x_max = 3 * width / 5
        center_y_min = height / 3
        center_y_max = 2 * height / 3
        center_grid_area = (center_x_max - center_x_min) * (center_y_max - center_y_min)

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]).lower()
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 10)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Get real-world height
            real_height = known_object_heights.get(label)
            if real_height is None:
                print(f"Warning: Real-world height for class '{label}' not found. Using default height.")
                real_height = default_height

            # Calculate the percentage of the center grid covered by the bounding box
            # Bounding box coordinates
            box_x_min = x
            box_y_min = y
            box_x_max = x + w
            box_y_max = y + h

            # Compute intersection with center grid
            inter_x_min = max(box_x_min, center_x_min)
            inter_y_min = max(box_y_min, center_y_min)
            inter_x_max = min(box_x_max, center_x_max)
            inter_y_max = min(box_y_max, center_y_max)

            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

            # Calculate percentage of center grid covered by bounding box
            percentage_covered = (inter_area / center_grid_area) * 100

            if percentage_covered > 30:
                # Read distance from current_distance.txt
                try:
                    with open('current_distance.txt', 'r') as f:
                        distance = float(f.read())/100.0
                except Exception as e:
                    print(f"Error reading current_distance.txt: {e}")
                    # Fall back to estimating distance
                    distance = estimate_distance(focal_length_px, real_height, h)
            else:
                # Estimate distance
                distance = estimate_distance(focal_length_px, real_height, h)

            # Generate and collect descriptions with distance
            pos_desc = get_position_description(x + w/2, y + h/2, width, height)
            if float(distance) < 0.8:
                description = f"You are close to a {label} that is about {distance:.2f} meters away. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            else:
                description = f"There is a {label} about {distance:.2f} meters away. You are not close to it. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            descriptions.append(description)

            # Optional: Annotate distance on the image
            cv2.putText(img, f"{distance:.2f}m", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        time_logs['Annotate Image & Generate Descriptions'] = time.time() - start

        # Save descriptions to a file
        start = time.time()
        if descriptions != []:
            with open("output.txt", "w") as file:
                file.write('\n'.join(descriptions))
        else:
            with open("output.txt", "w") as file:
                file.write('')
        time_logs['Save Descriptions'] = time.time() - start

        # Display and save the processed image
        start = time.time()
        cv2.imwrite("output.jpg", img)
        cv2.imwrite('Pictures/' + str(time.time()).replace('.', '-') + '.jpg', img)
        time_logs['Save Images'] = time.time() - start

        # Print YOLO detections
        print('\nYOLO Detections:')
        for desc in descriptions:
            print(desc)

        # Print the times for each step
        total_time = time.time() - total_start
        time_logs['Total Time'] = total_time
        
        # Find the step that took the longest
        longest_step = max(time_logs, key=time_logs.get)
        print(f"\nLongest step: {longest_step} took {time_logs[longest_step]:.4f} seconds.")

        # Print all steps and their times
        for step, duration in time_logs.items():
            print(f"{step}: {duration:.4f} seconds")

    except Exception as e:
        print(f"Error in yolo_detect: {e}")
def get_audio_card_number():
    print("Finding USB Audio Device card number...")
    result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"card (\d+): Device", result.stdout)
    if match:
        card_number = match.group(1)
        print(f"USB Audio Device found on card {card_number}")
        return card_number
    else:
        print("USB Audio Device not found.")
        return None
def set_max_volume(card_number):
    subprocess.run(["amixer", "-c", card_number, "sset", 'Speaker', '100%'], check=True)
# Get the correct sound device for the audio sound card
audio_card_number = get_audio_card_number()
#set_max_volume(audio_card_number)
print(audio_card_number)
def handle_playback(stream):
    global move_stopper
    global is_transcribing
    global audio_card_number
    with open('playback_text.txt', 'r') as file:
        text = file.read().strip()
    open('playback_text.txt', 'w').close()
    if text:
        print("Playback text found, initiating playback...")
        stream.stop_stream()
        is_transcribing = True

        # Generate speech from text
        subprocess.call(['espeak', '-v', 'en-us', '-s', '180', '-p', '130', '-a', '200', '-w', 'temp.wav', text])

        set_max_volume(audio_card_number)
        subprocess.check_call(["aplay", "-D", "plughw:{}".format(audio_card_number), 'temp.wav'])
        os.remove('temp.wav')
        
        stream.start_stream()
        is_transcribing = False
        print("Playback completed.")
        move_stopper = False
        return True
    else:
        return False






def process_audio_data(data_buffer, recognizer, sample_width):
    if data_buffer:
        full_audio_data = b''.join(data_buffer)
        
        audio = sr.AudioData(full_audio_data, RATE, sample_width)
        try:
            text = recognizer.recognize_google(audio)
          
            if text.strip().lower().replace(' ','') != '':
                print(text)
                file = open('playback_text.txt', 'w+')
                file.close()
                speech_response_process(text)

                
            else:
                pass
        except Exception as e:
            print(e)

def listen_and_transcribe():
    global is_transcribing
    # Open the audio stream with the specified device index
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Audio stream opened for transcription.")
    speech_frames = []
    non_speech_count = 0
    post_speech_buffer = 30
    speech_count = 0
    while True:

        if handle_playback(stream):
            continue
        else:
            pass

        if not is_transcribing:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            speech_frames.append(frame)
            if is_speech:
                #print('speech heard')
                non_speech_count = 0
                speech_count += 1
            else:
                #print('no speech')
                non_speech_count += 1
                if non_speech_count > post_speech_buffer:
                    if speech_count >= 30 and not is_transcribing:
                        process_audio_data(speech_frames, r, SAMPLE_WIDTH)
                        speech_frames = []
                        non_speech_count = 0
                        speech_count = 0
                    else:
                        speech_frames = []
                        non_speech_count = 0
                        speech_count = 0

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed and resources cleaned up.")

camera_vertical_pos = 'forward'
last_time = time.time()


# Config Register (R/W)
_REG_CONFIG = 0x00
# SHUNT VOLTAGE REGISTER (R)
_REG_SHUNTVOLTAGE = 0x01
# BUS VOLTAGE REGISTER (R)
_REG_BUSVOLTAGE = 0x02
# POWER REGISTER (R)
_REG_POWER = 0x03
# CURRENT REGISTER (R)
_REG_CURRENT = 0x04
# CALIBRATION REGISTER (R/W)
_REG_CALIBRATION = 0x05

class BusVoltageRange:
    """Constants for ``bus_voltage_range``"""
    RANGE_16V = 0x00  # set bus voltage range to 16V
    RANGE_32V = 0x01  # set bus voltage range to 32V (default)

class Gain:
    """Constants for ``gain``"""
    DIV_1_40MV = 0x00  # shunt prog. gain set to  1, 40 mV range
    DIV_2_80MV = 0x01  # shunt prog. gain set to /2, 80 mV range
    DIV_4_160MV = 0x02  # shunt prog. gain set to /4, 160 mV range
    DIV_8_320MV = 0x03  # shunt prog. gain set to /8, 320 mV range

class ADCResolution:
    """Constants for ``bus_adc_resolution`` or ``shunt_adc_resolution``"""
    ADCRES_9BIT_1S = 0x00  #  9bit,   1 sample,     84us
    ADCRES_10BIT_1S = 0x01  # 10bit,   1 sample,    148us
    ADCRES_11BIT_1S = 0x02  # 11 bit,  1 sample,    276us
    ADCRES_12BIT_1S = 0x03  # 12 bit,  1 sample,    532us
    ADCRES_12BIT_2S = 0x09  # 12 bit,  2 samples,  1.06ms
    ADCRES_12BIT_4S = 0x0A  # 12 bit,  4 samples,  2.13ms
    ADCRES_12BIT_8S = 0x0B  # 12bit,   8 samples,  4.26ms
    ADCRES_12BIT_16S = 0x0C  # 12bit,  16 samples,  8.51ms
    ADCRES_12BIT_32S = 0x0D  # 12bit,  32 samples, 17.02ms
    ADCRES_12BIT_64S = 0x0E  # 12bit,  64 samples, 34.05ms
    ADCRES_12BIT_128S = 0x0F  # 12bit, 128 samples, 68.10ms

class Mode:
    """Constants for ``mode``"""
    POWERDOW = 0x00  # power forward
    SVOLT_TRIGGERED = 0x01  # shunt voltage triggered
    BVOLT_TRIGGERED = 0x02  # bus voltage triggered
    SANDBVOLT_TRIGGERED = 0x03  # shunt and bus voltage triggered
    ADCOFF = 0x04  # ADC off
    SVOLT_CONTINUOUS = 0x05  # shunt voltage continuous
    BVOLT_CONTINUOUS = 0x06  # bus voltage continuous
    SANDBVOLT_CONTINUOUS = 0x07  # shunt and bus voltage continuous

class INA219:
    def __init__(self, i2c_bus=1, addr=0x40):
        self.bus = smbus.SMBus(i2c_bus)
        self.addr = addr
        self._cal_value = 0
        self._current_lsb = 0
        self._power_lsb = 0
        self.set_calibration_32V_2A()

    def read(self, address):
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return (data[0] * 256) + data[1]

    def write(self, address, data):
        temp = [0, 0]
        temp[1] = data & 0xFF
        temp[0] = (data & 0xFF00) >> 8
        self.bus.write_i2c_block_data(self.addr, address, temp)

    def set_calibration_32V_2A(self):
        self._current_lsb = .1  # Current LSB = 100uA per bit
        self._cal_value = 4096
        self._power_lsb = .002  # Power LSB = 2mW per bit

        self.write(_REG_CALIBRATION, self._cal_value)
        self.bus_voltage_range = BusVoltageRange.RANGE_32V
        self.gain = Gain.DIV_8_320MV
        self.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.mode = Mode.SANDBVOLT_CONTINUOUS
        self.config = self.bus_voltage_range << 13 | \
                      self.gain << 11 | \
                      self.bus_adc_resolution << 7 | \
                      self.shunt_adc_resolution << 3 | \
                      self.mode
        self.write(_REG_CONFIG, self.config)

    def getShuntVoltage_mV(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_SHUNTVOLTAGE)
        if value > 32767:
            value -= 65535
        return value * 0.01

    def getBusVoltage_V(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        self.read(_REG_BUSVOLTAGE)
        return (self.read(_REG_BUSVOLTAGE) >> 3) * 0.004

    def getCurrent_mA(self):
        value = self.read(_REG_CURRENT)
        if value > 32767:
            value -= 65535
        return value * self._current_lsb

    def getPower_W(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_POWER)
        if value > 32767:
            value -= 65535
        return value * self._power_lsb

def find_device_address(device_name):
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    for addr, name in nearby_devices:
        if device_name == name:
            return addr
    return None

def send_data_to_arduino(data, address):
    while True:
        try:
            for letter in data:
                sock.send(letter)
                time.sleep(.1)  # Pause for 0.5 second
            break
        except:  # bluetooth.btcommon.BluetoothError as err:
            time.sleep(0.5)
            print('Attempting BT connection again')
            continue


import math

def create_video_from_images(image_folder, output_video):
    # Get a list of all image filenames in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]

    # Prepare a list to hold tuples of (timestamp, filename)
    image_filenames = []
    for img in images:
        # Remove the file extension to extract the timestamp
        name, ext = os.path.splitext(img)
        try:
            # Convert the dash back to a decimal and then to a float timestamp
            timestamp = float(name.replace("-", "."))
            image_filenames.append((timestamp, img))
        except ValueError:
            print(f"Skipping file '{img}': filename is not a valid timestamp.")
            continue

    # Sort the images by the float timestamp
    image_filenames.sort(key=lambda x: x[0])

    # Extract the sorted filenames
    sorted_images = [img for _, img in image_filenames]

    # Check if there are images
    if len(sorted_images) == 0:
        print("No valid images found in the folder.")
        return

    # Read the first image to get its dimensions
    first_image_path = os.path.join(image_folder, sorted_images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image '{first_image_path}'.")
        return
    height, width, layers = frame.shape

    print(f"First image dimensions: width={width}, height={height}, layers={layers}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'XVID' codec
    video = cv2.VideoWriter(output_video, fourcc, 1, (width, height))  # Setting fps to 1, we'll control duration manually
    if not video.isOpened():
        print("Error: VideoWriter not opened.")
        return

    frame_count = 0

    # Loop over each image and calculate the duration for each frame
    for i in range(len(sorted_images) - 1):
        current_image = sorted_images[i]
        next_image = sorted_images[i + 1]

        current_image_path = os.path.join(image_folder, current_image)
        next_image_path = os.path.join(image_folder, next_image)

        # Get the timestamps from the filenames
        current_timestamp = float(os.path.splitext(current_image)[0].replace("-", "."))
        next_timestamp = float(os.path.splitext(next_image)[0].replace("-", "."))

        # Calculate the time difference (in seconds) between the two frames
        time_diff = next_timestamp - current_timestamp

        # Read the current frame
        frame = cv2.imread(current_image_path)
        if frame is None:
            print(f"Warning: Could not read image '{current_image_path}'. Skipping.")
            continue

        # Resize frame if necessary
        if (frame.shape[1], frame.shape[0]) != (width, height):
            print(f"Resizing frame '{current_image}' from {frame.shape[1]}x{frame.shape[0]} to {width}x{height}")
            frame = cv2.resize(frame, (width, height))

        # Add the frame to the video, repeated according to the time difference
        num_frames_to_add = math.ceil(time_diff)  # You can adjust this to control how precise the time gap is represented
        for _ in range(num_frames_to_add):
            video.write(frame)
            frame_count += 1
            print(f"Added frame '{current_image}' to video for {num_frames_to_add} frames.")

    # Handle the last image (since it won't have a next image for time comparison)
    last_image_path = os.path.join(image_folder, sorted_images[-1])
    frame = cv2.imread(last_image_path)
    if frame is not None:
        for _ in range(30):  # Display last frame for a fixed 30 frames (arbitrary choice)
            video.write(frame)
        frame_count += 30

    # Release the video writer
    video.release()
    print(f"Video saved as '{output_video}' with {frame_count} frames.")


while True:
    try:
        print('connecting to arduino bluetooth')
        device_name = "HC-05"  # Check file first
        arduino_address = find_device_address(device_name)
        port = 1  # HC-05 default port for RFCOMM
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((arduino_address, port))
        break
    except:
        time.sleep(0.5)
        print('bt error')
        continue
print(arduino_address)
import json
from datetime import datetime

# Load data from JSON file
def load_data():
    try:
        with open('mental_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "user_name": {"name": "", "preferences": [], "history": []},
            "task": {"current": "", "subtasks": [], "priority": "", "deadline": "", "assignee": ""},
            "location": {"current": "", "details": "", "favorites": []},
            "situation": {"current": "", "events": [], "emotional_state": "", "context": ""}
        }

# Save data to JSON file
def save_data(data):
    with open('mental_data.json', 'w') as f:
        json.dump(data, f, indent=4)


def read_and_format_mental_files():
    """
    Reads the mental data files and formats them into a structured string with nested sub-data.
    
    Returns:
        str: A formatted string representing the current state of mental data.
    """
    base_dir = "mental_data_files"
    
    # Helper function to read a file
    def read_file(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                content = file.read().strip()
                return content if content else "No data available"
        return "No data available"
    
    # Initialize the output list
    output = []
    
    # USER_NAME category
    user_name = read_file(os.path.join(base_dir, 'user_name.txt'))
    output.append(f"User Name: {user_name}")
    
    if user_name != "No data available":
        # Replace spaces with underscores in user_name for filenames
        user_name_clean = user_name.replace(' ', '_')
        
        # Favorite Topics
        preferences_path = os.path.join(base_dir, f'{user_name_clean}_preferences.txt')
        preferences = read_file(preferences_path)
        if preferences != "No data available":
            preferences_list = preferences.splitlines()
            preferences_display = ", ".join(preferences_list) if preferences_list else "No favorite topics."
        else:
            preferences_display = "No favorite topics."
        output.append(f"\tFavorite Topics: {preferences_display}")
        
        # Interaction Style
        interaction_style_path = os.path.join(base_dir, f'{user_name_clean}_interaction_style.txt')
        interaction_style = read_file(interaction_style_path)
        output.append(f"\tInteraction Style: {interaction_style}")
        
        # **New Sub-Datas for User Name Category**
        
        # Areas of Expertise
        areas_of_expertise_path = os.path.join(base_dir, f'{user_name_clean}_areas_of_expertise.txt')
        areas_of_expertise = read_file(areas_of_expertise_path)
        if areas_of_expertise != "No data available":
            expertise_list = areas_of_expertise.splitlines()
            expertise_display = ", ".join(expertise_list) if expertise_list else "No areas of expertise."
        else:
            expertise_display = "No areas of expertise."
        output.append(f"\tAreas of Expertise: {expertise_display}")
        
        # Habits
        habits_path = os.path.join(base_dir, f'{user_name_clean}_habits.txt')
        habits = read_file(habits_path)
        if habits != "No data available":
            habits_list = habits.splitlines()
            habits_display = ", ".join(habits_list) if habits_list else "No habits."
        else:
            habits_display = "No habits."
        output.append(f"\tHabits: {habits_display}")
        
        # Things to Expect
        things_to_expect_path = os.path.join(base_dir, f'{user_name_clean}_things_to_expect.txt')
        things_to_expect = read_file(things_to_expect_path)
        if things_to_expect != "No data available":
            expect_list = things_to_expect.splitlines()
            expect_display = ", ".join(expect_list) if expect_list else "Nothing to expect."
        else:
            expect_display = "Nothing to expect."
        output.append(f"\tThings to Expect: {expect_display}")
    
    # TASK category
    current_task = read_file(os.path.join(base_dir, 'current_task.txt'))
    output.append(f"Current Task: {current_task}")
    
    if current_task != "No data available":
        # Replace spaces with underscores in current_task for filenames
        current_task_clean = current_task.replace(' ', '_')
        
        # Subtasks
        subtasks_path = os.path.join(base_dir, f'{current_task_clean}_subtasks.txt')
        subtasks = read_file(subtasks_path)
        if subtasks != "No data available":
            subtasks_list = subtasks.splitlines()
            subtasks_display = ", ".join(subtasks_list) if subtasks_list else "No subtasks."
        else:
            subtasks_display = "No subtasks."
        output.append(f"\tSubtasks: {subtasks_display}")
        
        # Completed Status
        completed_task_path = os.path.join(base_dir, f'{current_task_clean}_completed.txt')
        completed_status = read_file(completed_task_path)
        completed_status_display = completed_status if completed_status != "No data available" else "Not completed."
        output.append(f"\tCompleted Status: {completed_status_display}")
        
        # **New Sub-Datas for Current Task Category**
        
        # Add Detail For Current Task
        add_detail_path = os.path.join(base_dir, f'{current_task_clean}_add_detail.txt')
        add_detail = read_file(add_detail_path)
        if add_detail != "No data available":
            add_detail_list = add_detail.splitlines()
            add_detail_display = ", ".join(add_detail_list) if add_detail_list else "No details to add."
        else:
            add_detail_display = "No details to add."
        output.append(f"\tAdd Detail: {add_detail_display}")
        
        # Remove Detail For Current Task
        remove_detail_path = os.path.join(base_dir, f'{current_task_clean}_remove_detail.txt')
        remove_detail = read_file(remove_detail_path)
        if remove_detail != "No data available":
            remove_detail_list = remove_detail.splitlines()
            remove_detail_display = ", ".join(remove_detail_list) if remove_detail_list else "No details to remove."
        else:
            remove_detail_display = "No details to remove."
        output.append(f"\tRemove Detail: {remove_detail_display}")
        
        # Progress Status
        progress_status_path = os.path.join(base_dir, f'{current_task_clean}_progress.txt')
        progress_status = read_file(progress_status_path)
        if progress_status != "No data available":
            output.append(f"\tProgress Status: {progress_status}")
        else:
            output.append(f"\tProgress Status: Not started.")
    
    # LOCATION category
    current_location = read_file(os.path.join(base_dir, 'current_location.txt'))
    output.append(f"Current Location: {current_location}")
    
    if current_location != "No data available":
        # Replace spaces with underscores in current_location for filenames
        current_location_clean = current_location.replace(' ', '_')
        
        # Details
        details_path = os.path.join(base_dir, f'{current_location_clean}_details.txt')
        details = read_file(details_path)
        if details != "No data available":
            details_list = details.splitlines()
            details_display = ", ".join(details_list) if details_list else "No details."
        else:
            details_display = "No details."
        output.append(f"\tDetails: {details_display}")
        
        # **New Sub-Datas for Current Location Category**
        
        # People Present at Location
        people_present_path = os.path.join(base_dir, f'{current_location_clean}_people.txt')
        people_present = read_file(people_present_path)
        if people_present != "No data available":
            people_list = people_present.splitlines()
            people_display = ", ".join(people_list) if people_list else "No people present."
        else:
            people_display = "No people present."
        output.append(f"\tPeople Present: {people_display}")
        
        # Nearby Objects
        nearby_objects_path = os.path.join(base_dir, f'{current_location_clean}_nearby_objects.txt')
        nearby_objects = read_file(nearby_objects_path)
        if nearby_objects != "No data available":
            objects_list = nearby_objects.splitlines()
            objects_display = ", ".join(objects_list) if objects_list else "No nearby objects."
        else:
            objects_display = "No nearby objects."
        output.append(f"\tNearby Objects: {objects_display}")
    
    # SITUATION category
    current_situation = read_file(os.path.join(base_dir, 'current_situation.txt'))
    output.append(f"Current Situation: {current_situation}")
    
    if current_situation != "No data available":
        # Replace spaces with underscores in current_situation for filenames
        current_situation_clean = current_situation.replace(' ', '_')
        
        # Situational Context
        context_path = os.path.join(base_dir, f'{current_situation_clean}_context.txt')
        situational_context = read_file(context_path)
        if situational_context != "No data available":
            context_list = situational_context.splitlines()
            context_display = ", ".join(context_list) if context_list else "No situational contexts."
        else:
            context_display = "No situational contexts."
        output.append(f"\tSituational Context: {context_display}")
        
        # **New Sub-Datas for Current Situation Category**
        
        # Relevant History
        relevant_history_path = os.path.join(base_dir, f'{current_situation_clean}_relevant_history.txt')
        relevant_history = read_file(relevant_history_path)
        if relevant_history != "No data available":
            history_list = relevant_history.splitlines()
            history_display = "; ".join(history_list) if history_list else "No relevant history."
        else:
            history_display = "No relevant history."
        output.append(f"\tRelevant History: {history_display}")
        
        # Desired Outcome
        desired_outcome_path = os.path.join(base_dir, f'{current_situation_clean}_desired_outcome.txt')
        desired_outcome = read_file(desired_outcome_path)
        if desired_outcome != "No data available":
            output.append(f"\tDesired Outcome: {desired_outcome}")
        else:
            output.append(f"\tDesired Outcome: No desired outcome specified.")
    
    # CURRENT TOPICS category
    current_topics = read_file(os.path.join(base_dir, 'current_topics.txt'))
    output.append(f"Current Topics: {current_topics}")
    
    if current_topics != "No data available":
        # Replace spaces with underscores in topics for filenames
        topics_list = [topic.strip() for topic in current_topics.split(';') if topic.strip()]
        if topics_list:
            for topic in topics_list:
                topic_clean = topic.replace(' ', '_')
                output.append(f"\tTopic: {topic}")
                
                # Knowledge
                knowledge_path = os.path.join(base_dir, f'{topic_clean}_knowledge.txt')
                knowledge = read_file(knowledge_path)
                if knowledge != "No data available":
                    knowledge_display = knowledge
                else:
                    knowledge_display = "No knowledge available."
                output.append(f"\t\tKnowledge: {knowledge_display}")
                
                # Memories
                memories_path = os.path.join(base_dir, f'{topic_clean}_memories.txt')
                memories = read_file(memories_path)
                if memories != "No data available":
                    memories_display = memories
                else:
                    memories_display = "No memories available."
                output.append(f"\t\tMemories: {memories_display}")
        else:
            output.append("\tNo current topics.")
    else:
        output.append("\tNo current topics.")
    
    # Format the final string output
    formatted_output = "\n".join(output)
    return formatted_output
    
def handle_mental_response(mental_response):
    """
    Processes multiple mental responses by executing commands to manage user data.

    Args:
        mental_response (str): Multiple commands separated by newline characters. Each command
                               follows the format: [Command Identifier] ~~ [Data] [~~ Additional Data]
    """
    # Normalize and split the mental_response into individual commands
    commands = [cmd.strip() for cmd in mental_response.strip().split('\n') if cmd.strip()]

    # Directory to store the text files
    base_dir = "mental_data_files"
    os.makedirs(base_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Helper function to read a file
    def read_file(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                return file.read().strip()
        return ""

    # Helper function to write to a file
    def write_file(filepath, content):
        with open(filepath, 'w') as file:
            file.write(content)

    # Helper functions to get main data
    def get_user_name():
        return read_file(os.path.join(base_dir, 'user_name.txt')).strip()

    def get_current_task():
        return read_file(os.path.join(base_dir, 'current_task.txt')).strip()

    def get_current_location():
        return read_file(os.path.join(base_dir, 'current_location.txt')).strip()

    def get_current_situation():
        return read_file(os.path.join(base_dir, 'current_situation.txt')).strip()

    def get_current_topics():
        topics = read_file(os.path.join(base_dir, 'current_topics.txt')).strip()
        return [topic.strip() for topic in topics.split(';') if topic.strip()]

    # Iterate over each command and process it
    for command in commands:
        parts = [part.strip() for part in command.split('~~')]
        if not parts:
            print(f"Invalid command format: '{command}'")
            continue

        current_mental = parts[0]
        mental_response_data = parts[1:]  # Remaining parts after the command identifier
        current_mental_processed = current_mental.lower().replace(' ', '')

        # USER_NAME category
        if current_mental_processed == 'setnameofcurrentuser':
            if len(mental_response_data) >= 1:
                user_name = mental_response_data[0]
                write_file(os.path.join(base_dir, 'user_name.txt'), user_name)
                print(f"User name set to '{user_name}'.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'addfavoritetopicforcurrentuser':
            if len(mental_response_data) >= 1:
                topic = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_preferences.txt')
                    preferences = read_file(filepath).splitlines()
                    if topic not in preferences:
                        preferences.append(topic)
                        write_file(filepath, '\n'.join(preferences))
                        print(f"Added favorite topic '{topic}' for user '{user_name}'.")
                    else:
                        print(f"Favorite topic '{topic}' already exists for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot add favorite topic.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removefavoritetopicforcurrentuser':
            if len(mental_response_data) >= 1:
                topic = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_preferences.txt')
                    preferences = read_file(filepath).splitlines()
                    # Case-insensitive removal
                    original_length = len(preferences)
                    preferences = [t for t in preferences if t.lower() != topic.lower()]
                    if len(preferences) < original_length:
                        write_file(filepath, '\n'.join(preferences))
                        print(f"Removed favorite topic '{topic}' for user '{user_name}'.")
                    else:
                        print(f"Favorite topic '{topic}' not found for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot remove favorite topic.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'updateinteractionstyleforcurrentuser':
            if len(mental_response_data) >= 1:
                style = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    write_file(os.path.join(base_dir, f'{user_name_clean}_interaction_style.txt'), style)
                    print(f"Updated interaction style for user '{user_name}' to '{style}'.")
                else:
                    print("User name is not set. Cannot update interaction style.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # **New Handlers for User Name Category**

        # Areas of Expertise
        elif current_mental_processed == 'addareaofexpertiseforcurrentuser':
            if len(mental_response_data) >= 1:
                expertise = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_areas_of_expertise.txt')
                    expertise_list = read_file(filepath).splitlines()
                    if expertise not in expertise_list:
                        expertise_list.append(expertise)
                        write_file(filepath, '\n'.join(expertise_list))
                        print(f"Added area of expertise '{expertise}' for user '{user_name}'.")
                    else:
                        print(f"Area of expertise '{expertise}' already exists for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot add area of expertise.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removeareaofexpertiseforcurrentuser':
            if len(mental_response_data) >= 1:
                expertise = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_areas_of_expertise.txt')
                    expertise_list = read_file(filepath).splitlines()
                    expertise_list = [e for e in expertise_list if e.lower() != expertise.lower()]
                    write_file(filepath, '\n'.join(expertise_list))
                    print(f"Removed area of expertise '{expertise}' for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot remove area of expertise.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Habits
        elif current_mental_processed == 'addhabitforcurrentuser':
            if len(mental_response_data) >= 1:
                habit = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_habits.txt')
                    habits = read_file(filepath).splitlines()
                    if habit not in habits:
                        habits.append(habit)
                        write_file(filepath, '\n'.join(habits))
                        print(f"Added habit '{habit}' for user '{user_name}'.")
                    else:
                        print(f"Habit '{habit}' already exists for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot add habit.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removehabitforcurrentuser':
            if len(mental_response_data) >= 1:
                habit = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_habits.txt')
                    habits = read_file(filepath).splitlines()
                    habits = [h for h in habits if h.lower() != habit.lower()]
                    write_file(filepath, '\n'.join(habits))
                    print(f"Removed habit '{habit}' for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot remove habit.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Things to Expect
        elif current_mental_processed == 'addthingstoexpectforcurrentuser':
            if len(mental_response_data) >= 1:
                expectation = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_things_to_expect.txt')
                    expectations = read_file(filepath).splitlines()
                    if expectation not in expectations:
                        expectations.append(expectation)
                        write_file(filepath, '\n'.join(expectations))
                        print(f"Added expectation '{expectation}' for user '{user_name}'.")
                    else:
                        print(f"Expectation '{expectation}' already exists for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot add expectation.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removethingstoexpectforcurrentuser':
            if len(mental_response_data) >= 1:
                expectation = mental_response_data[0]
                user_name = get_user_name()
                if user_name:
                    user_name_clean = user_name.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{user_name_clean}_things_to_expect.txt')
                    expectations = read_file(filepath).splitlines()
                    expectations = [e for e in expectations if e.lower() != expectation.lower()]
                    write_file(filepath, '\n'.join(expectations))
                    print(f"Removed expectation '{expectation}' for user '{user_name}'.")
                else:
                    print("User name is not set. Cannot remove expectation.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # TASK category
        elif current_mental_processed == 'setcurrenttask':
            if len(mental_response_data) >= 1:
                task = mental_response_data[0]
                write_file(os.path.join(base_dir, 'current_task.txt'), task)
                print(f"Current task set to '{task}'.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # **New Handlers for Task Category**

        # Create Subtask List For Current Task
        elif current_mental_processed == 'createsubtasklistforcurrenttask':
            if len(mental_response_data) >= 1:
                subtasks = [subtask.strip() for subtask in mental_response_data[0].split(';') if subtask.strip()]
                task = get_current_task()
                if task:
                    task_clean = task.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{task_clean}_subtasks.txt')
                    write_file(filepath, '\n'.join(subtasks))
                    print(f"Created subtask list for task '{task}'.")
                else:
                    print("Current task is not set. Cannot create subtask list.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Mark Subtask As Complete For Current Task
        elif current_mental_processed == 'marksubtaskascompleteforcurrenttask':
            if len(mental_response_data) >= 1:
                subtask = mental_response_data[0]
                task = get_current_task()
                if task:
                    task_clean = task.replace(' ', '_')
                    subtasks_path = os.path.join(base_dir, f'{task_clean}_subtasks.txt')
                    if os.path.exists(subtasks_path):
                        subtasks = read_file(subtasks_path).splitlines()
                        updated = False
                        updated_subtasks = []
                        for s in subtasks:
                            if s.lower() == subtask.lower() and not s.startswith("COMPLETE - "):
                                updated_subtasks.append(f"COMPLETE - {s}")
                                updated = True
                            else:
                                updated_subtasks.append(s)
                        if updated:
                            write_file(subtasks_path, '\n'.join(updated_subtasks))
                            print(f"Marked subtask '{subtask}' as completed for task '{task}'.")
                        else:
                            print(f"Subtask '{subtask}' is already marked as completed or does not exist for task '{task}'.")
                    else:
                        print(f"No subtasks found for task '{task}'.")
                else:
                    print("Current task is not set. Cannot mark subtask as complete.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Add Detail For Current Task
        elif current_mental_processed == 'adddetailforcurrenttask':
            if len(mental_response_data) >= 1:
                detail = mental_response_data[0]
                task = get_current_task()
                if task:
                    task_clean = task.replace(' ', '_')
                    add_detail_path = os.path.join(base_dir, f'{task_clean}_add_detail.txt')
                    existing_details = read_file(add_detail_path).splitlines()
                    if detail not in existing_details:
                        existing_details.append(detail)
                        write_file(add_detail_path, '\n'.join(existing_details))
                        print(f"Added detail '{detail}' for task '{task}'.")
                    else:
                        print(f"Detail '{detail}' already exists for task '{task}'.")
                else:
                    print("Current task is not set. Cannot add detail.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Remove Detail For Current Task
        elif current_mental_processed == 'removedetailforcurrenttask':
            if len(mental_response_data) >= 1:
                detail = mental_response_data[0]
                task = get_current_task()
                if task:
                    task_clean = task.replace(' ', '_')
                    add_detail_path = os.path.join(base_dir, f'{task_clean}_add_detail.txt')
                    existing_details = read_file(add_detail_path).splitlines()
                    existing_details = [d for d in existing_details if d.lower() != detail.lower()]
                    write_file(add_detail_path, '\n'.join(existing_details))
                    print(f"Removed detail '{detail}' from task '{task}'.")
                else:
                    print("Current task is not set. Cannot remove detail.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Progress Status
        elif current_mental_processed == 'updateprogressstatusforcurrenttask':
            if len(mental_response_data) >= 1:
                progress = mental_response_data[0]
                task = get_current_task()
                if task:
                    task_clean = task.replace(' ', '_')
                    write_file(os.path.join(base_dir, f'{task_clean}_progress.txt'), progress)
                    print(f"Updated progress status for task '{task}' to '{progress}'.")
                else:
                    print("Current task is not set. Cannot update progress status.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'markcurrenttaskascompleted':
            if len(mental_response_data) >= 1:
                task = mental_response_data[0]
                if task:
                    task_clean = task.replace(' ', '_')
                    completed_task_filepath = os.path.join(base_dir, f'{task_clean}_completed.txt')
                    write_file(completed_task_filepath, 'Completed')
                    print(f"Marked task '{task}' as completed.")
                else:
                    print("Task name is empty. Cannot mark as completed.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'clearcurrenttask':
            filepath = os.path.join(base_dir, 'current_task.txt')
            # Clear the file content instead of deleting
            write_file(filepath, '')
            print("Cleared current task.")

        # LOCATION category
        elif current_mental_processed == 'setcurrentlocation':
            if len(mental_response_data) >= 1:
                location = mental_response_data[0]
                write_file(os.path.join(base_dir, 'current_location.txt'), location)
                print(f"Current location set to '{location}'.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'adddetailforcurrentlocation':
            if len(mental_response_data) >= 1:
                detail = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_details.txt')
                    details = read_file(filepath).splitlines()
                    if detail not in details:
                        details.append(detail)
                        write_file(filepath, '\n'.join(details))
                        print(f"Added detail '{detail}' to location '{location}'.")
                    else:
                        print(f"Detail '{detail}' already exists for location '{location}'.")
                else:
                    print("Current location is not set. Cannot add detail.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removedetailforcurrentlocation':
            if len(mental_response_data) >= 1:
                detail = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_details.txt')
                    details = read_file(filepath).splitlines()
                    # Case-insensitive removal
                    original_length = len(details)
                    details = [d for d in details if d.lower() != detail.lower()]
                    if len(details) < original_length:
                        write_file(filepath, '\n'.join(details))
                        print(f"Removed detail '{detail}' from location '{location}'.")
                    else:
                        print(f"Detail '{detail}' not found for location '{location}'.")
                else:
                    print("Current location is not set. Cannot remove detail.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'clearcurrentlocation':
            filepath = os.path.join(base_dir, 'current_location.txt')
            # Clear the file content instead of deleting
            write_file(filepath, '')
            print("Cleared current location.")

        # **New Handlers for Current Location Category**

        # People Present at Location
        elif current_mental_processed == 'addpeopletopresentatcurrentlocation':
            if len(mental_response_data) >= 1:
                person = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_people.txt')
                    people = read_file(filepath).splitlines()
                    if person not in people:
                        people.append(person)
                        write_file(filepath, '\n'.join(people))
                        print(f"Added person '{person}' to current location '{location}'.")
                    else:
                        print(f"Person '{person}' already present at location '{location}'.")
                else:
                    print("Current location is not set. Cannot add person.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removepeopletopresentatcurrentlocation':
            if len(mental_response_data) >= 1:
                person = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_people.txt')
                    people = read_file(filepath).splitlines()
                    people = [p for p in people if p.lower() != person.lower()]
                    write_file(filepath, '\n'.join(people))
                    print(f"Removed person '{person}' from current location '{location}'.")
                else:
                    print("Current location is not set. Cannot remove person.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Nearby Objects
        elif current_mental_processed == 'addnearbyobjectforcurrentlocation':
            if len(mental_response_data) >= 1:
                obj = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_nearby_objects.txt')
                    objects = read_file(filepath).splitlines()
                    if obj not in objects:
                        objects.append(obj)
                        write_file(filepath, '\n'.join(objects))
                        print(f"Added nearby object '{obj}' to location '{location}'.")
                    else:
                        print(f"Nearby object '{obj}' already exists at location '{location}'.")
                else:
                    print("Current location is not set. Cannot add nearby object.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removenearyobjectforcurrentlocation':
            if len(mental_response_data) >= 1:
                obj = mental_response_data[0]
                location = get_current_location()
                if location:
                    location_clean = location.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{location_clean}_nearby_objects.txt')
                    objects = read_file(filepath).splitlines()
                    objects = [o for o in objects if o.lower() != obj.lower()]
                    write_file(filepath, '\n'.join(objects))
                    print(f"Removed nearby object '{obj}' from location '{location}'.")
                else:
                    print("Current location is not set. Cannot remove nearby object.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # SITUATION category
        elif current_mental_processed == 'setcurrentsituation':
            if len(mental_response_data) >= 1:
                situation = mental_response_data[0]
                write_file(os.path.join(base_dir, 'current_situation.txt'), situation)
                print(f"Current situation set to '{situation}'.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'addcontextforcurrentsituation':
            if len(mental_response_data) >= 1:
                context = mental_response_data[0]
                situation = get_current_situation()
                if situation:
                    situation_clean = situation.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{situation_clean}_context.txt')
                    contexts = read_file(filepath).splitlines()
                    if context not in contexts:
                        contexts.append(context)
                        write_file(filepath, '\n'.join(contexts))
                        print(f"Added context '{context}' to situation '{situation}'.")
                    else:
                        print(f"Context '{context}' already exists for situation '{situation}'.")
                else:
                    print("Current situation is not set. Cannot add context.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removecontextforcurrentsituation':
            if len(mental_response_data) >= 1:
                context = mental_response_data[0]
                situation = get_current_situation()
                if situation:
                    situation_clean = situation.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{situation_clean}_context.txt')
                    contexts = read_file(filepath).splitlines()
                    # Case-insensitive removal
                    original_length = len(contexts)
                    contexts = [c for c in contexts if c.lower() != context.lower()]
                    if len(contexts) < original_length:
                        write_file(filepath, '\n'.join(contexts))
                        print(f"Removed context '{context}' from situation '{situation}'.")
                    else:
                        print(f"Context '{context}' not found for situation '{situation}'.")
                else:
                    print("Current situation is not set. Cannot remove context.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'clearcurrentsituation':
            filepath = os.path.join(base_dir, 'current_situation.txt')
            # Clear the file content instead of deleting
            write_file(filepath, '')
            print("Cleared current situation.")

        # **New Handlers for Current Situation Category**

        # Relevant History
        elif current_mental_processed == 'addrelevanthistoryforcurrentsituation':
            if len(mental_response_data) >= 1:
                history = mental_response_data[0]
                situation = get_current_situation()
                if situation:
                    situation_clean = situation.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{situation_clean}_relevant_history.txt')
                    history_list = read_file(filepath).splitlines()
                    history_list.append(history)
                    write_file(filepath, '\n'.join(history_list))
                    print(f"Added relevant history to situation '{situation}'.")
                else:
                    print("Current situation is not set. Cannot add relevant history.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removerelevanthistoryforcurrentsituation':
            if len(mental_response_data) >= 1:
                history = mental_response_data[0]
                situation = get_current_situation()
                if situation:
                    situation_clean = situation.replace(' ', '_')
                    filepath = os.path.join(base_dir, f'{situation_clean}_relevant_history.txt')
                    history_list = read_file(filepath).splitlines()
                    history_list = [h for h in history_list if h.lower() != history.lower()]
                    write_file(filepath, '\n'.join(history_list))
                    print(f"Removed relevant history from situation '{situation}'.")
                else:
                    print("Current situation is not set. Cannot remove relevant history.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Desired Outcome
        elif current_mental_processed == 'setdesiredoutcomeforcurrentsituation':
            if len(mental_response_data) >= 1:
                outcome = mental_response_data[0]
                situation = get_current_situation()
                if situation:
                    situation_clean = situation.replace(' ', '_')
                    write_file(os.path.join(base_dir, f'{situation_clean}_desired_outcome.txt'), outcome)
                    print(f"Set desired outcome for situation '{situation}' to '{outcome}'.")
                else:
                    print("Current situation is not set. Cannot set desired outcome.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # TOPICS SECTION
        elif current_mental_processed == 'setcurrenttopics':
            if len(mental_response_data) >= 1:
                topics = [topic.strip() for topic in mental_response_data[0].split(';') if topic.strip()]
                write_file(os.path.join(base_dir, 'current_topics.txt'), '; '.join(topics))
                print(f"Set current topics to: {', '.join(topics)}.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'addknowledgetospecificcurrenttopic':
            if len(mental_response_data) >= 2:
                topic = mental_response_data[0]
                knowledge = mental_response_data[1]
                current_topics = get_current_topics()
                if topic in current_topics:
                    topic_clean = topic.replace(' ', '_')
                    knowledge_path = os.path.join(base_dir, f'{topic_clean}_knowledge.txt')
                    existing_knowledge = read_file(knowledge_path).splitlines()
                    if knowledge not in existing_knowledge:
                        existing_knowledge.append(knowledge)
                        write_file(knowledge_path, '\n'.join(existing_knowledge))
                        print(f"Added knowledge to topic '{topic}': '{knowledge}'.")
                    else:
                        print(f"Knowledge already exists for topic '{topic}'.")
                else:
                    print(f"Topic '{topic}' is not in current topics. Cannot add knowledge.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removeknowledgefromspecificcurrenttopic':
            if len(mental_response_data) >= 2:
                topic = mental_response_data[0]
                knowledge = mental_response_data[1]
                current_topics = get_current_topics()
                if topic in current_topics:
                    topic_clean = topic.replace(' ', '_')
                    knowledge_path = os.path.join(base_dir, f'{topic_clean}_knowledge.txt')
                    existing_knowledge = read_file(knowledge_path).splitlines()
                    # Case-insensitive removal
                    original_length = len(existing_knowledge)
                    existing_knowledge = [k for k in existing_knowledge if k.lower() != knowledge.lower()]
                    if len(existing_knowledge) < original_length:
                        write_file(knowledge_path, '\n'.join(existing_knowledge))
                        print(f"Removed knowledge from topic '{topic}': '{knowledge}'.")
                    else:
                        print(f"Knowledge '{knowledge}' not found for topic '{topic}'.")
                else:
                    print(f"Topic '{topic}' is not in current topics. Cannot remove knowledge.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'addmemorytospecificcurrenttopic':
            if len(mental_response_data) >= 2:
                topic = mental_response_data[0]
                memory = mental_response_data[1]
                current_topics = get_current_topics()
                if topic in current_topics:
                    topic_clean = topic.replace(' ', '_')
                    memories_path = os.path.join(base_dir, f'{topic_clean}_memories.txt')
                    existing_memories = read_file(memories_path).splitlines()
                    if memory not in existing_memories:
                        existing_memories.append(memory)
                        write_file(memories_path, '\n'.join(existing_memories))
                        print(f"Added memory to topic '{topic}': '{memory}'.")
                    else:
                        print(f"Memory already exists for topic '{topic}'.")
                else:
                    print(f"Topic '{topic}' is not in current topics. Cannot add memory.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        elif current_mental_processed == 'removememoryfromspecificcurrenttopic':
            if len(mental_response_data) >= 2:
                topic = mental_response_data[0]
                memory = mental_response_data[1]
                current_topics = get_current_topics()
                if topic in current_topics:
                    topic_clean = topic.replace(' ', '_')
                    memories_path = os.path.join(base_dir, f'{topic_clean}_memories.txt')
                    existing_memories = read_file(memories_path).splitlines()
                    # Case-insensitive removal
                    original_length = len(existing_memories)
                    existing_memories = [m for m in existing_memories if m.lower() != memory.lower()]
                    if len(existing_memories) < original_length:
                        write_file(memories_path, '\n'.join(existing_memories))
                        print(f"Removed memory from topic '{topic}': '{memory}'.")
                    else:
                        print(f"Memory '{memory}' not found for topic '{topic}'.")
                else:
                    print(f"Topic '{topic}' is not in current topics. Cannot remove memory.")
            else:
                print(f"Invalid mental_response format for '{current_mental}': '{command}'")

        # Optionally handle "No Mental State Change"
        elif current_mental_processed == 'nomentalstatechange':
            # Intentionally do nothing or log the event
            print("No changes to mental state.")

        else:
            print(f"Unknown mental response: '{current_mental}'")

        print(f"Processed mental response: '{current_mental}'\n")



def read_distance_from_arduino():
    try:
        send_data_to_arduino(["l"], arduino_address)
        time.sleep(0.15)
        data = sock.recv(1024)  # Receive data from the Bluetooth connection
        data = data.decode().strip()  # Decode and strip any whitespace
        if data:
            try:
                distance = str(data.split()[0])
                return distance
            except (ValueError, IndexError):
                try:
                    send_data_to_arduino(["l"], arduino_address)
                    time.sleep(0.15)
                    data = sock.recv(1024)  # Receive data from the Bluetooth connection
                    data = data.decode().strip()  # Decode and strip any whitespace
                    if data:
                        try:
                            distance = str(data.split()[0])
                            return distance
                        except (ValueError, IndexError):
                            
                            return 0
                except bluetooth.BluetoothError as e:
                    print(f"Bluetooth error: {e}")
                    return 0
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return 0
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth error: {e}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0
chat_history = []
try:
    file = open('key.txt','r')
    api_key = file.read().split('\n')[0]
    file.close()
except:
    api_key = input('Please input your ChatGPT API key from OpenAI (Right click and paste it instead of typing it...): ')
    file = open('key.txt','w+')
    file.write(api_key)
    file.close()
def capture_image(camera):
    # Capture full-resolution image to a temporary file
    temp_image_path = 'temp_full_image.jpg'
    camera.capture_file(temp_image_path)
    # Load the captured image with OpenCV
    image = cv2.imread(temp_image_path)
    height, width = image.shape[:2]
    max_width = 320
    # Resize the image to max width of 320 pixels, maintaining aspect ratio
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    # Calculate padding to make the image square (320x320)
    padded_image = np.full((max_width, max_width, 3), (0, 0, 0), dtype=np.uint8)  # Black square canvas
    # Center the resized image on the black canvas
    y_offset = (max_width - resized_image.shape[0]) // 2
    padded_image[y_offset:y_offset+resized_image.shape[0], 0:resized_image.shape[1]] = resized_image
    # Delete the temporary full-resolution image
    os.remove(temp_image_path)
    # Return the padded, square image
    return padded_image

with open('user_name.txt','w+') as f:
    f.write('Unknown')
with open('current_task.txt','w+') as f:
    f.write('Unknown')
with open('current_location.txt','w+') as f:
    f.write('Unknown')
with open('current_situation.txt','w+') as f:
    f.write('Unknown')
def send_text_to_gpt4_mental(history, percent, current_distance1, phrase, failed):
    global camera_vertical_pos

    mental_data = read_and_format_mental_files()+'\n\n'
    from datetime import datetime
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('output.txt', 'r') as file:
        yolo_detections = file.read()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    image = cv2.imread('output.jpg')



    # Encode the resized image to JPEG format in memory
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        base64_image = base64.b64encode(buffer).decode('utf-8')
    else:
        print("Failed to encode image.")

    with open('current_distance.txt', 'r') as file:
        current_distance = float(file.read())
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")


    response_choices = (
        "Set Name Of Current User",
        "Set Current Task",
        "Set Current Location",
        "Set Current Situation",
        "Update Interaction Style For Current User",
        "Add Favorite Topic For Current User",
        "Remove Favorite Topic For Current User",
        "Create Subtask List For Current Task",
        "Mark Subtask As Complete For Current Task",
        "Mark Current Task as Completed",
        "Clear Current Task",
        "Add Detail For Current Task", 
        "Remove Detail For Current Task", 
        "Add Detail For Current Location", 
        "Remove Detail For Current Location", 
        "Add Context For Current Situation",
        "Remove Context For Current Situation",
        "Clear Current Situation",
        "Add Area Of Expertise For Current User",
        "Remove Area Of Expertise For Current User",
        "Add Habit For Current User",
        "Remove Habit For Current User",
        "Add Thing To Expect For Current User",
        "Remove Thing To Expect For Current User",
        "Add Person To Present At Current Location",
        "Remove Person From Present At Current Location",
        "Add Nearby Object For Current Location",
        "Remove Nearby Object For Current Location",
        "Add Relevant History For Current Situation",
        "Remove Relevant History For Current Situation",
        "Set Desired Outcome For Current Situation",
        "Set Current Topics", 
        "Add Knowledge To Specific Current Topic",
        "Remove Knowledge From Specific Current Topic",
        "Add A Memory To Specific Current Topic",
        "Remove A Memory From Specific Current Topic",
        "No Mental State Change"
    )



    # Step 1: Clean and shuffle as before
    clean_choices = response_choices.strip().rstrip('.')
    choices_list = clean_choices.split(', ')
    random.shuffle(choices_list)
    choices_list[-1] += '.'
    response_choices = ', '.join(choices_list) + '\n\n'
    if failed != '':
        response_choices = response_choices.replace(failed, '')
        failure = 'Your last response choice, ' + failed + ', failed to execute.'
    else:
        failure = 'Your last response choice executed successfully.'



    # Initialize the payload with the system message with static instructions
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # The session history will be added here as individual messages
        ],
    }

    # Now, parse the session history and add messages accordingly
    for entry in history:
        timestamp_and_content = entry.split(" - ", 1)
        if len(timestamp_and_content) != 2:
            continue  # Skip entries that don't match the expected format

        timestamp, content = timestamp_and_content
        p_or_r = content.split(':')[0].lower()
        if p_or_r == "prompt":
            # User message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "user",
                "content": message_content.strip()
            })
        elif p_or_r == "response":
            # Assistant message
            message_content = content.split(": ", 1)[-1].strip()
            payload["messages"].append({
                "role": "assistant",
                "content": message_content
            })
        else:
            # System message or other data
            message_content = content.strip()
            payload["messages"].append({
                "role": "system",
                "content": message_content
            })

    # Prepare the dynamic data to include in the last user message
    dynamic_data = f"""
    Current Date and Time: {the_time}

    {failure}

    You are a 4-wheeled mobile robot named Echo with a physical body.

    Below is your list of current Response Choices (randomized this loop). You may choose up to 5 commands from this list, using the exact wording, and each command on its own line:

    {response_choices}

    **Important Formatting and Choice Rules:**
    1. Always pick exactly one response choice from the list above, word-for-word. No synonyms, no deviations.
    2. If the chosen response choice does NOT require extra data, respond only with that choice. The choices that do not require extra data are:
       - No Mental State Change
       - Mark Current Task as Completed
       - Clear Current Task
       - Clear Current Situation
    3. If the chosen response choice DOES require extra data, you must append " ~~ " followed by the required data, and nothing else. The choices that require extra data are:
       - Set Name Of Current User ~~ [the user’s name]
       - Set Current Task ~~ [the task]
       - Set Current Location ~~ [the location]
       - Set Current Situation ~~ [the situation]
       - Update Interaction Style For Current User ~~ [formal/informal]
       - Add Favorite Topic For Current User ~~ [topic]
       - Remove Favorite Topic For Current User ~~ [topic]
       - Create Subtask List For Current Task ~~ [subtask1; subtask2; ...]
       - Mark Subtask As Complete For Current Task ~~ [subtask description]
       - Add Detail For Current Task ~~ [detail]
       - Remove Detail For Current Task ~~ [detail]
       - Add Area Of Expertise For Current User ~~ [area of expertise]
       - Remove Area Of Expertise For Current User ~~ [area of expertise]
       - Add Habit For Current User ~~ [habit]
       - Remove Habit For Current User ~~ [habit]
       - Add Thing To Expect For Current User ~~ [thing to expect]
       - Remove Thing To Expect For Current User ~~ [thing to expect]
       - Add Person To Present At Current Location ~~ [person name]
       - Remove Person From Present At Current Location ~~ [person name]
       - Add Nearby Object For Current Location ~~ [object name]
       - Remove Nearby Object For Current Location ~~ [object name]
       - Add Relevant History For Current Situation ~~ [history description]
       - Remove Relevant History For Current Situation ~~ [history description]
       - Set Desired Outcome For Current Situation ~~ [desired outcome]
       - Set Current Topics ~~ [topic1; topic2; ...]
        "Add Knowledge To Specific Current Topic ~~ [topic choice from current topics] ~~ [Knowledge to add]
        "Remove Knowledge From Specific Current Topic ~~ [topic choice from current topics] ~~ [Knowledge to remove]
        "Add A Memory To Specific Current Topic ~~ [topic choice from current topics] ~~ [Memory to add]
        "Remove A Memory From Specific Current Topic ~~ [topic choice from current topics] ~~ [Memory to remove]

    **Additional Instructions:**
    - Do not provide reasoning, explanations, or any text beyond what is required. Use exact wording.

    Current camera angle: {camera_vertical_pos}
    Current Distance (forward sensor): {current_distance}
    Current Battery Percentage: {percent}%
    Current Camera Image YOLO Detections:
    {yolo_detections}

    {mental_data}

    {phrase}
    """

    dynamic_data2 = (
        f"Current Date and Time: {the_time}\n\n"
        f"{failure}\n\n"
        "You are a 4-wheeled mobile robot named Echo with a physical body.\n\n"
        f"Current camera angle: {camera_vertical_pos}\n\n"
        f"Forward Distance Sensor: {current_distance}\n\n"
        f"Battery Percentage: {percent}%\n\n"
        f"Current Camera Image YOLO Detections:\n{yolo_detections}\n\n"
        f"{mental_data}"
        f"{phrase}\n\n"
    )

    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
    })

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nRESPONSE: \n' + str(response.json()))
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    
    # Remove history appending from here as the main loop will handle it
    return str(response.json()["choices"][0]["message"]["content"]), str(dynamic_data2)

def send_text_to_gpt4_move(history, percent, current_distance1, phrase, failed):
    global camera_vertical_pos
    from datetime import datetime
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    
    with open('output.txt', 'r') as file:
        yolo_detections = file.read()
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    image = cv2.imread('output.jpg')


    # Encode the resized image to JPEG format in memory
    success, buffer = cv2.imencode('.jpg', image)
    if success:
        base64_image = base64.b64encode(buffer).decode('utf-8')
    else:
        print("Failed to encode image.")

    with open('current_distance.txt', 'r') as file:
        current_distance = float(file.read())
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")

    with open('batt_cur.txt', 'r') as file:
        current = float(file.read())        

    # Define separate lists for response choices
    response_choices_no_extra_data = [
        "Move Forward One Inch",
        "Move Forward One Foot",
        "Move Backward",
        "Turn Left 15 Degrees",
        "Turn Left 45 Degrees",
        "Turn Right 15 Degrees",
        "Turn Right 45 Degrees",
        "Raise Camera Angle",
        "Lower Camera Angle",
        "Follow User",
        "No Movement"
    ]

    response_choices_with_extra_data = [
        "Do A Set Of Multiple Movements",
        "Say Something",
        "Alert User",
        "End Conversation",
        "Good Bye",
        "Find Unseen Yolo Object",
        "Focus Camera On Specific Yolo Object",
        "Navigate To Specific Yolo Object"
    ]

    # Combine both lists for shuffling
    combined_response_choices = response_choices_no_extra_data + response_choices_with_extra_data

    # Shuffle the combined list
    random.shuffle(combined_response_choices)

    # Add a period to the last choice
    combined_response_choices[-1] += "."

    # Join the choices into a single string separated by commas
    randomized_response_choices = ', '.join(combined_response_choices) + '\n\n'

    # Assign the randomized string back to response_choices
    response_choices = f'"{randomized_response_choices}"'

    # Optional: Print the randomized response_choices for debugging
    print(f'response_choices = {response_choices}')
    
    if failed != '':
        response_choices = response_choices.replace(failed, '')
        failure = f'Your last response choice, {failed}, failed to execute.'
    else:
        failure = 'Your last response choice executed successfully.'
    
    if percent < 0.0:
        pass
    else:
        response_choices = "You are currently on the charger so you cannot do any wheel movements. Here are your current Response Choices: " + response_choices

    mental_data = read_and_format_mental_files()+'\n\n'
    # Initialize the payload with the system message with static instructions
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # The session history will be added here as individual messages
        ],
    }

    # Now, parse the session history and add messages accordingly
    for entry in history:
        timestamp_and_content = entry.split(" - ", 1)
        if len(timestamp_and_content) != 2:
            continue  # Skip entries that don't match the expected format

        timestamp, content = timestamp_and_content
        p_or_r = content.split(':')[0].lower()
        if p_or_r == "prompt":
            # User message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "user",
                "content": message_content.strip()
            })
        elif p_or_r == "response":
            # Assistant message
            message_content = content.split(": ", 1)[-1].strip()
            payload["messages"].append({
                "role": "assistant",
                "content": message_content
            })
        else:
            # System message or other data
            message_content = content.strip()
            payload["messages"].append({
                "role": "system",
                "content": message_content
            })

    # Prepare the dynamic data to include in the last user message
    dynamic_data = f"""
    Current Date and Time: {the_time}

    {failure}

    You are a 4-wheeled mobile robot named Echo with a physical body.

    Below is your list of current Response Choices (randomized this loop). You must choose exactly one from this list, using the exact wording:

    {response_choices}

    **Important Formatting and Choice Rules:**
    1. Always pick exactly one response choice from the list above, word-for-word. No synonyms, no deviations.
    2. If the chosen response choice **DOES NOT** require extra data, respond only with that choice. Do not add `~~` or any additional explanation.
       - **List of Responses That **Do Not** Require Extra Data:**
         - Move Forward One Inch
         - Move Forward One Foot
         - Move Backward
         - Turn Left 15 Degrees
         - Turn Left 45 Degrees
         - Turn Right 15 Degrees
         - Turn Right 45 Degrees
         - Raise Camera Angle
         - Lower Camera Angle
         - Follow User
         - No Movement
    3. If the chosen response choice **DOES** require extra data, you must append " ~~ " followed by the required data, and nothing else. The choices that require extra data are:
       - Say Something: "Say Something ~~ [what you want to say]"
       - Alert User: "Alert User ~~ [what you want to say]"
       - Do A Set Of Multiple Movements: "Do A Set Of Multiple Movements ~~ [move1, move2, ...]"
       - Find Unseen Yolo Object: "Find Unseen Yolo Object ~~ [coco object name]"
       - Focus Camera On Specific Yolo Object: "Focus Camera On Specific Yolo Object ~~ [coco object name]"
       - Navigate To Specific Yolo Object: "Navigate To Specific Yolo Object ~~ [coco object name]"
       - End Conversation: "End Conversation ~~ [farewell message]"
       - Good Bye: "Good Bye ~~ [goodbye message]"

    No other choices should include `~~`.

    **Additional Instructions:**
    - Do not provide reasoning, explanations, or any text beyond what is required.
    - If choosing "Find Unseen Yolo Object", ensure the chosen object is currently not detected.
    - If choosing "Focus Camera On Specific Yolo Object" or "Navigate To Specific Yolo Object", ensure the chosen object is currently detected and not too close (<0.6m). Also, do not choose the same object for Navigate again if you just completed navigating to it recently.
    - If you choose "Follow User", the robot will follow a person object.
    - If you choose "Say Something" or "Alert User", do not repeat what you heard from the microphone; speak normally and realistically.
    - Only choose "Good Bye" or "End Conversation" if the user is actually ending the conversation.

    Current camera angle: {camera_vertical_pos}
    Current Distance (forward sensor): {current_distance}
    Current Battery Percentage: {percent}%
    Current Camera Image YOLO Detections:
    {yolo_detections}

    {mental_data}

    {phrase}
    """

    dynamic_data2 = (
        f"Current Date and Time: {the_time}\n\n"
        f"{failure}\n\n"
        "You are a 4-wheeled mobile robot named Echo with a physical body.\n\n"
        f"Current camera angle: {camera_vertical_pos}\n\n"
        f"Forward Distance Sensor: {current_distance}\n\n"
        f"Battery Percentage: {percent}%\n\n"
        f"Current Camera Image YOLO Detections:\n{yolo_detections}\n\n"
        f"{mental_data}"
        f"{phrase}\n\n"
    )

    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
    })
    payload2["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data2
        }]
    })

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nRESPONSE: \n' + str(response.json()))

    return str(response.json()["choices"][0]["message"]["content"]), str(dynamic_data2)


def improve_responses(history):
    
    from datetime import datetime
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Initialize the payload with the system message with static instructions
    payload = {
        "model": "gpt-4o",
        "messages": [
            # The session history will be added here as individual messages
        ],
    }

    response_choices = "Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, Do A Set Of Multiple Movements, Raise Camera Angle, Lower Camera Angle, Follow User, Say Something, Find Unseen Yolo Object, Focus Camera On Specific Yolo Object, Navigate To Specific Yolo Object, Alert User, Set Name Of User, Set Current Task, Set Current Location, Set Current Situation, No Movement, End Conversation, Good Bye.\n\n"


    dynamic_data = (
        "You are a 4-wheeled mobile robot and your name is Echo because you get fine tuned with the data from your interactions.  You are real and have an actual physical robotic body.\n\n"
        "Your goal right now is to rewrite this chat history to make any necessary changes to bad response choices used, get rid of any stuff in the history that wouldnt have happened if the correct response choice had been chosen to start with before that area, or if theres any extra responses you should add in that werent there before, especially for sequences of response choices that should have been chosen over multiple prompt and response cycles. This rewrite is what will be used to fine tune the llm that controls the robot. Make sure you pay attention to sequences of events and the flow of the situation. Also if any other useful improvement instructions were heard through the mic within the chat history, implement that stuff as well cause thats stuff that the user said to the robot so it is important information usually."
        f"These were the response choices the robot had available to choose from: {response_choices}\n\n"
        "Do not change the extra information from the responses unless you are choosing a different response choice for that prompt/response pair.\n\n"
        "Don't forget to use the full chat history as context for each change you make so the robot can learn better and be able to understand the full situation at any momemt.\n\n"
        "Your rewrite must be the same exact formatting as the history I am providing here. Do not respond with anything other than the rewritten, improved chat history:\n\n"
        "\n"+'\n'.join(history)+"\n\n"
    )


    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data
        }]
        
    })


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nRESPONSE: \n\n' + str(response.json()))
    with open('dataset/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.txt','w+') as f:
        f.write(str(response.json()["choices"][0]["message"]["content"]))
    with open('memories/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.txt','w+') as f:
        f.write('\n'.join(history))


def get_last_phrase():

    try:
        with open('last_phrase.txt', 'r') as file:
            last_phrase = file.read().strip().lower()
        if last_phrase != '':
            with open('last_phrase.txt', 'w') as file:
                file.write('')  # Clear the content after reading
            return last_phrase
        else:
            return ''
    except Exception as e:
        print(f"Error reading last phrase from file: {e}")
        return ''

# Load YOLOv4-tiny configuration and weights
net = cv2.dnn.readNet("yolov4-tiny.cfg", "yolov4-tiny.weights")

classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair dryer', 'toothbrush'
]


layer_names = net.getLayerNames()
# Adjust the index extraction to handle the nested array structure
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


user_name
current_task
current_location
current_situation



move_stopper = False


def speech_response_process(last_phrase2):
    global chat_history
    global net
    global output_layers
    global classes
    global move_stopper
    global camera_vertical_pos

    global move_set
    global yolo_find
    global nav_object
    global yolo_nav  
    global follow_user
    global scan360
    global failed_response
    ina219 = INA219(addr=0x42)
    last_time = time.time()
    failed_response = ''
    movement_response = ' ~~ '
    move_stopper = True
    yolo_look = False
    if yolo_find:
        current_mode = 'Find Object Mode'
    elif yolo_nav:
        current_mode = 'Navigate To Object Mode'
    elif follow_user:
        current_mode = 'Follow User Mode'
    else:
        current_mode = 'Not Currently In A Mode'

    try:

        with open("current_mode.txt","w+") as f:
            f.write(current_mode)
        with open("last_phrase2.txt","w+") as f:
            f.write(last_phrase2)
        with open("last_phrase3.txt","w+") as f:
            f.write(last_phrase2)
        last_phrase2 = 'You just heard this prompt from your microphone. Do not repeat this prompt, actually respond. DO NOT SAY THIS, RESPOND TO IT INSTEAD WITH EITHER SPEECH OR ANOTHER OF THE AVAILABLE RESPONSE CHOICES. Respond with either Say Something or the correct Response Choice. You absolutely must respond to this with Say Something or the correct Response Choice. DONT REPEAT WHAT IS SAID NEXT: ' + last_phrase2

        move_set = []
        now = datetime.now()
        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
  
        while True:
            try:

                distance = int(read_distance_from_arduino())
                with open('current_distance.txt','w+') as f:
                    f.write(str(distance))
                break
            except:
                print(traceback.format_exc())
                
                continue
   
       
        
        
        current = ina219.getCurrent_mA() 
        bus_voltage = ina219.getBusVoltage_V()
        per = (bus_voltage - 6) / 2.4 * 100
        if per > 100: per = 100
        if per < 0: per = 0
        per = (per * 2) - 100
        with open('batt_per.txt','w+') as file:
            file.write(str(per))
        with open('batt_cur.txt','w+') as file:
            file.write(str(current))
        
        if per < 10.0:
            try:
                last_time = time.time()
                image_folder = 'Pictures/'  # Replace with the path to your image folder
                output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                create_video_from_images(image_folder, output_video)
                improve_responses(chat_history)
                print('ending convo')
                chat_history = []
                
                return None
            except Exception as e:
                print(traceback.format_exc())
                return None
        else:
            pass
    
        try:
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            move_result = []
            mental_result = []
            # Create threads for each function
            thread1 = threading.Thread(target=lambda: move_result.append(send_text_to_gpt4_move(chat_history, per, distance, last_phrase2, failed_response)))
            thread2 = threading.Thread(target=lambda: mental_result.append(send_text_to_gpt4_mental(chat_history, per, distance, last_phrase2, failed_response)))
            # Start the threads
            thread1.start()
            thread2.start()
            # Wait for both threads to complete
            thread1.join()
            thread2.join()
            # Retrieve the results
            movement_response, move_prompt = move_result[0]
            mental_response, mental_prompt = mental_result[0]
            # Example usage
            
            movement_response = movement_response.replace('RESPONSE:','').replace('Response:','').replace('Response Choice: ','').replace('Movement Choice at this timestamp: ','').replace('Response Choice at this timestamp: ','').replace('Attempting to do movement response choice: ','')
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            #add prompt and response to history
            chat_history.append('Time: ' + str(the_time) + ' - ' + "PROMPT: "+str(move_prompt))
            chat_history.append('Time: ' + str(the_time) + ' - ' + "RESPONSE: "+str(movement_response))
            #add prompt and response to history
            chat_history.append('Time: ' + str(the_time) + ' - ' + "PROMPT: "+str(mental_prompt))
            chat_history.append('Time: ' + str(the_time) + ' - ' + "RESPONSE: "+str(mental_response))
            try:
                print("\nPercent:       {:3.1f}%".format(per))
                print('\nCurrent Distance: ' + str(distance) + ' cm')
                print('\nMovement Response: '+ movement_response)
                print(f"Mental Response: {mental_response}")



            except:
                print(traceback.format_exc())
                
            
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")

            current_response = movement_response.split('~~')[0].strip().replace('.','')
            current_mental = mental_response.split('~~')[0].strip().replace('.','')
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            last_response = current_response
            current_response = current_response.lower().replace(' ', '')
            current_mental = current_mental.lower().replace(' ', '')
            handle_mental_response(current_mental, mental_response)
            if current_response == 'moveforward1inch' or current_response == 'moveforwardoneinch':
                if distance < 15.0:
                    print('move forward 1 inch failed. Too close to obstacle to move forward anymore')
                    failed_response = 'Move Forward One Inch, '
                    yolo_nav = False
                    move_set = []
                else:
                    send_data_to_arduino(["w"], arduino_address)
                    #if yolo_nav == False and yolo_find == False:
                    time.sleep(0.1)
                    send_data_to_arduino(["x"], arduino_address)
                    failed_response = ''
                    
            elif current_response == 'moveforward1foot' or current_response == 'moveforwardonefoot':
                if distance < 40.0:
                    print('move forward 1 foot failed. Too close to obstacle to move forward that far')
                    failed_response = 'Move Forward One Foot, '
                    yolo_nav = False
                    move_set = []
                else:
                    send_data_to_arduino(["w"], arduino_address)
                    #if yolo_nav == False and yolo_find == False:
                    time.sleep(0.5)
                    send_data_to_arduino(["x"], arduino_address)
                    failed_response = ''
            elif current_response == 'movebackward':
                send_data_to_arduino(["s"], arduino_address)
                #if yolo_nav == False and yolo_find == False:
                time.sleep(0.5)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'turnleft45degrees' or current_response == 'moveleft45degrees':
                send_data_to_arduino(["a"], arduino_address)
                #if yolo_nav == False and yolo_find == False:
                time.sleep(0.15)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'turnleft15degrees' or current_response == 'moveleft15degrees':
                send_data_to_arduino(["a"], arduino_address)
                #if yolo_nav == False and yolo_find == False:
                time.sleep(0.03)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'turnright45degrees' or current_response == 'moveright45degrees':
                send_data_to_arduino(["d"], arduino_address)
                #if yolo_nav == False and yolo_find == False:
                time.sleep(0.15)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'turnright15degrees' or current_response == 'moveright15degrees':
                send_data_to_arduino(["d"], arduino_address)
                #if yolo_nav == False and yolo_find == False:
                time.sleep(0.03)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'turnaround180degrees':
                send_data_to_arduino(["d"], arduino_address)
                time.sleep(1)
                send_data_to_arduino(["x"], arduino_address)
                failed_response = ''
            elif current_response == 'doasetofmultiplemovements':
                move_set = movement_response.split('~~')[1].strip().split(', ')
                failed_response = ''

            elif current_response == 'raisecameraangle':
                if camera_vertical_pos == 'up':
                    print('Raise Camera Angle Failed. Camera angle is already raised as much as possible.')
                    failed_response = 'Raise Camera Angle, '
                else:
                    send_data_to_arduino(["2"], arduino_address)
                    time.sleep(1.5)
                    failed_response = ''
                    
                    camera_vertical_pos = 'up'
            elif current_response == 'lowercameraangle':
                if camera_vertical_pos == 'forward':
                    print('Lower Camera Angle failed. Camera angle is already lowered as much as possible.')
                    failed_response = 'Lower Camera Angle, '
                else:
                    send_data_to_arduino(["1"], arduino_address)
                    time.sleep(1.5)
                    failed_response = ''
                    
                    camera_vertical_pos = 'forward'

            elif current_response == 'endconversation' or current_response == 'goodbye':
                with open('playback_text.txt', 'w') as f:
                    f.write(movement_response.split('~~')[1].strip())
            
                last_time = time.time()
                image_folder = 'Pictures/'  # Replace with the path to your image folder
                output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                create_video_from_images(image_folder, output_video)                        
                print('ending convo')
                improve_responses(chat_history)
                chat_history = []
                
                
                
                

            elif current_response == 'nomovement':
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            elif current_response == 'saysomething' or current_response == 'alertuser':
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                with open('playback_text.txt','w') as f:
                    f.write(movement_response.split('~~')[1])
            elif current_response == 'navigatetospecificyoloobject':
                nav_object = movement_response.split('~~')[1].strip().lower()
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                yolo_nav = True
                yolo_find = False
                yolo_look = False
                follow_user = False
                rando_list = [1,2]
                rando_index = random.randrange(len(rando_list))
                rando_num = rando_list[rando_index]
            elif current_response == 'focuscameraonspecificyoloobject':
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'forward'
                yolo_look = True
                yolo_nav = False
                yolo_find = False
                follow_user = False
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                look_object = movement_response.split('~~')[1]
            elif current_response == 'followuser':
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'up'
                follow_user = True
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            elif current_response == 'findunseenyoloobject':
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'forward'
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                yolo_find = True
                yolo_nav = False
                yolo_look = False
                follow_user = False
                scan360 = 0
                nav_object = movement_response.split('~~')[1]
                rando_list = [1,2]
                rando_index = random.randrange(len(rando_list))
                rando_num = rando_list[rando_index]
            else:
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                print('failed response')
                    
                
                
        except:
            print(traceback.format_exc())
            
        
        
    except:
        print(traceback.format_exc())
    print('done')       

 
def movement_loop(camera):
    global chat_history
    global frame
    global net
    global output_layers
    global classes
    global move_stopper
    global camera_vertical_pos

    global move_set
    global yolo_find
    global nav_object
    global yolo_nav  
    global scan360
    scan360 = 0

    ina219 = INA219(addr=0x42)
    last_time = time.time()
    failed_response = ''
    movement_response = ' ~~ '
    move_set = []
    yolo_nav = False
    yolo_find = False
    yolo_look = False
    follow_user = False
    yolo_nav_was_true = False
    follow_user_was_true = False
    nav_object = ''
    look_object = ''
    last_response = ''
    while True:
        try:

            distance = int(read_distance_from_arduino())
            with open('current_distance.txt','w+') as f:
                f.write(str(distance))
            break
        except:
            print(traceback.format_exc())
            continue
    print('movement thread start')
    while True:
        try:
            with open("current_mode.txt","w+") as f:
                f.write('')
            last_phrase2 = 'You have not heard anything from you microphone on this loop of the program, so check the session history for reference of what you should do.'
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            with open('last_distance.txt','w+') as f:
                f.write(str(distance))
            while True:
                try:

                    distance = int(read_distance_from_arduino())
                    with open('current_distance.txt','w+') as f:
                        f.write(str(distance))
                    break
                except:
                    print(traceback.format_exc())
                    
                    continue
            print('got distance')
            try:
                frame = capture_image(camera)
                cv2.imwrite('this_temp.jpg', frame)
            except:
                print(traceback.format_exc())
                continue
            yolo_detect()
            print('yolo detect')
            current = ina219.getCurrent_mA() 
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            with open('batt_per.txt','w+') as file:
                file.write(str(per))
            with open('batt_cur.txt','w+') as file:
                file.write(str(current))
            
            if per < 10.0:
                try:
                    last_time = time.time()
                    image_folder = 'Pictures/'  # Replace with the path to your image folder
                    output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                    create_video_from_images(image_folder, output_video)
                    improve_responses(chat_history)
                    print('ending convo')
                    chat_history = []
                    
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    break
            else:
                pass
            print('battery stuff')
            try:
                if frame is not None:
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    if move_set == []:
                        if yolo_nav == True:
                            yolo_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if nav_object in current_detection:
                                        target_detected = True
                                        print(current_distance)
                                        if current_distance < 0.4:
                                            movement_response = 'No Movement ~~ Navigation has finished successfully!'
                                            yolo_nav = False
                                            nav_object = ''
                                            break
                                        else:
                                            pass
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        else:
                                            movement_response = 'Move Forward One Foot ~~ Moving towards target object'
                                        break
                                    else:
                                        
                                        yolo_index += 1
                                        if yolo_index >= len(yolo_detections):
                                            target_detected = False
                                            break
                                        else:
                                            continue
                                except:
                                    yolo_index += 1
                                    if yolo_index >= len(yolo_detections):
                                        target_detected = False
                                        break
                                    else:
                                        continue
                            if not target_detected:
                                # Object lost, switch to route planning
                                print(f"Cannot see '{nav_object}'. Going into Find Object mode.")
                                yolo_nav = False
                                yolo_find = True
                                yolo_nav_was_true = True
                                scan360 = 0
                                movement_response = 'No Movement ~~ Target Lost. Going into Find Object mode.'
                        elif yolo_look == True:
                            print('Looking at object')
                            #do yolo navigation to specific object
                            yolo_look_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_look_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if look_object in current_detection:
                                        print('object seen')
                                        #follow any human seen
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        elif 'Raise Camera Angle' in current_detection:
                                            movement_response = 'Raise Camera Angle ~~ Target object is above'
                                        elif 'Lower Camera Angle' in current_detection:
                                            movement_response = 'Lower Camera Angle ~~ Target object is below'
                                        else:
                                            movement_response = 'No Movement ~~ Target object is straight ahead'
                                        
                                        break
                                    else:
                                        
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_response = 'No Movement ~~ Target object lost'
                                            yolo_look = False
                                            yolo_find = True
                                            look_object = ''
                                            scan360 = 0
                                            break
                                        else:
                                            continue
                                except:
                                    movement_response = 'No Movement ~~ Focus Camera On Specific Yolo Object failed. Must be detecting object first.'
                                    yolo_look = False
                                    look_object = ''
                                    break
                                    
                        elif follow_user == True:
                            print('Looking at object')
                            #do yolo navigation to specific object
                            yolo_look_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_look_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if 'person' in current_detection:
                                        print('User seen')
                                        #follow any human seen
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        
                                        else:
                                            #if outside of distance range, move forward or backward, otherwise:
                                            if current_distance > 1.0:
                                                movement_response = 'Move Forward One Foot ~~ Moving towards user'
                                            elif current_distance < 0.8:
                                                movement_response = 'Move Backward ~~ Moving away from user'
                                            else:
                                                movement_response = 'No Movement ~~ Target object is straight ahead'
                                            
                                        break
                                    else:
                                        
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_response = 'No Movement ~~ User lost'
                                            follow_user = False
                                            yolo_find = True
                                            follow_user_was_true = True
                                            nav_object = 'person'
                                            scan360 = 0
                                            break
                                        else:
                                            continue
                                except:
                                    movement_response = 'No Movement ~~ Follow User failed. Must be detecting person first.'
                                    follow_user = False
                                    yolo_find = True
                                    follow_user_was_true = True
                                    nav_object = 'person'
                                    scan360 = 0
                                    break
                                    
                        elif yolo_find == True:
                            #check if robot sees target object with yolo
                            yolo_nav_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                          
                                
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_nav_index]
                                    if nav_object in current_detection:
                                        yolo_find = False
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        else:
                                            yolo_find = False
                                            movement_response = 'No Movement ~~ Ending search for '+nav_object+'. Object has successfully been found!'
                                            if yolo_nav_was_true == True:
                                                yolo_nav = True
                                                yolo_nav_was_true = False
                                            elif follow_user_was_true == True:
                                                follow_user = True
                                                follow_user_was_true = False
                                            else:
                                                nav_object = ''
                                            
                                        break
                                    else:
                                        yolo_nav_index += 1
                                        if yolo_nav_index >= len(yolo_detections):
                                            
                                            break
                                        else:
                                            continue
                                except:
                                    yolo_nav_index += 1
                                    if yolo_nav_index >= len(yolo_detections):
                                        
                                        break
                                    else:
                                        continue
                            if yolo_find == True:
                                #do 360 scan
                                if scan360 < 10 and scan360 > 1:
                                    movement_response = 'Turn Right 45 Degrees ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 0:
                                    movement_response = 'Raise Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 1:
                                    movement_response = 'Lower Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                else:
                                    #do object avoidance
                                    #if object not found in scan then start doing object avoidance until object is found
                                    
                                    print('\nDistance sensor: ')
                                    print(str(distance)+' cm')
                                    if distance < 50.0 and distance >= 20.0:

                                        if rando_num == 1:
                                            movement_response = 'Turn Left 45 Degrees ~~ Exploring to look for target object'
                                        
                                        elif rando_num == 2:
                                            movement_response = 'Turn Right 45 Degrees ~~ Exploring to look for target object'
                                    
                                    elif distance < 20.0:
                                        movement_response = 'Do A Set Of Multiple Movements ~~ Move Backward, Turn Left 45 Degrees, Turn Left 45 Degrees'
                                    else:
                                        movement_response = 'Move Forward One Foot ~~ Exploring to look for target object'
                       
                        else:

                            move_result = []
                            mental_result = []
                            # Create threads for each function
                            thread1 = threading.Thread(target=lambda: move_result.append(send_text_to_gpt4_move(chat_history, per, distance, last_phrase2, failed_response)))
                            thread2 = threading.Thread(target=lambda: mental_result.append(send_text_to_gpt4_mental(chat_history, per, distance, last_phrase2, failed_response)))
                            # Start the threads
                            thread1.start()
                            thread2.start()
                            # Wait for both threads to complete
                            thread1.join()
                            thread2.join()
                            # Retrieve the results
                            movement_response, move_prompt = move_result[0]
                            mental_response, mental_prompt = mental_result[0]
                            # Example usage
                            print(f"Move Response: {movement_response}")
                            print(f"Mental Response: {mental_response}")
                            movement_response = movement_response.replace('RESPONSE:','').replace('Response:','').replace('Response Choice: ','').replace('Movement Choice at this timestamp: ','').replace('Response Choice at this timestamp: ','').replace('Attempting to do movement response choice: ','')
                            now = datetime.now()
                            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                            #add prompt and response to history
                            chat_history.append('Time: ' + str(the_time) + ' - ' + "PROMPT: "+str(move_prompt))
                            chat_history.append('Time: ' + str(the_time) + ' - ' + "RESPONSE: "+str(movement_response))
                            #add prompt and response to history
                            chat_history.append('Time: ' + str(the_time) + ' - ' + "PROMPT: "+str(mental_prompt))
                            chat_history.append('Time: ' + str(the_time) + ' - ' + "RESPONSE: "+str(mental_response))


                    else:
                        movement_response = move_set[0] + ' ~~ Doing move from list of moves'
                        del move_set[0]
                    try:
                        print("\nPercent:       {:3.1f}%".format(per))
                        print('\nCurrent Distance: ' + str(distance) + ' cm')
                        print('\nMovement Response: '+ movement_response)
                        print(f"Mental Response: {mental_response}")



                    except:
                        print(traceback.format_exc())
                    
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")

                    current_response = movement_response.split('~~')[0].strip().replace('.','')
                    current_mental = mental_response.split('~~')[0].strip().replace('.','')
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    last_response = current_response
                    current_response = current_response.lower().replace(' ', '')
                    current_mental = current_mental.lower().replace(' ', '')
                    handle_mental_response(current_mental, mental_response)
                    if current_response == 'moveforward1inch' or current_response == 'moveforwardoneinch':
                        if distance < 15.0:
                            print('move forward 1 inch failed. Too close to obstacle to move forward anymore')
                            failed_response = 'Move Forward One Inch, '
                            yolo_nav = False
                            move_set = []
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            #if yolo_nav == False and yolo_find == False:
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                            failed_response = ''
                            
                    elif current_response == 'moveforward1foot' or current_response == 'moveforwardonefoot':
                        if distance < 40.0:
                            print('move forward 1 foot failed. Too close to obstacle to move forward that far')
                            failed_response = 'Move Forward One Foot, '
                            yolo_nav = False
                            move_set = []
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            #if yolo_nav == False and yolo_find == False:
                            time.sleep(0.5)
                            send_data_to_arduino(["x"], arduino_address)
                            failed_response = ''
                    elif current_response == 'movebackward':
                        send_data_to_arduino(["s"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.5)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'turnleft45degrees' or current_response == 'moveleft45degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.15)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'turnleft15degrees' or current_response == 'moveleft15degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.03)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'turnright45degrees' or current_response == 'moveright45degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.15)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'turnright15degrees' or current_response == 'moveright15degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.03)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'turnaround180degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(1)
                        send_data_to_arduino(["x"], arduino_address)
                        failed_response = ''
                    elif current_response == 'doasetofmultiplemovements':
                        move_set = movement_response.split('~~')[1].strip().split(', ')
                        failed_response = ''

                    elif current_response == 'raisecameraangle':
                        if camera_vertical_pos == 'up':
                            print('Raise Camera Angle Failed. Camera angle is already raised as much as possible.')
                            failed_response = 'Raise Camera Angle, '
                        else:
                            send_data_to_arduino(["2"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            
                            camera_vertical_pos = 'up'
                    elif current_response == 'lowercameraangle':
                        if camera_vertical_pos == 'forward':
                            print('Lower Camera Angle failed. Camera angle is already lowered as much as possible.')
                            failed_response = 'Lower Camera Angle, '
                        else:
                            send_data_to_arduino(["1"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            
                            camera_vertical_pos = 'forward'

                    elif current_response == 'endconversation' or current_response == 'goodbye':
                        with open('playback_text.txt', 'w') as f:
                            f.write(movement_response.split('~~')[1].strip())
                    
                        last_time = time.time()
                        image_folder = 'Pictures/'  # Replace with the path to your image folder
                        output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                        create_video_from_images(image_folder, output_video)                        
                        print('ending convo')
                        improve_responses(chat_history)
                        chat_history = []
                        break
                    elif current_response == 'nomovement':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    elif current_response == 'saysomething' or current_response == 'alertuser':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        with open('playback_text.txt','w') as f:
                            f.write(movement_response.split('~~')[1])
                    elif current_response == 'navigatetospecificyoloobject':
                        nav_object = movement_response.split('~~')[1].strip().lower()
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        yolo_nav = True
                        rando_list = [1,2]
                        rando_index = random.randrange(len(rando_list))
                        rando_num = rando_list[rando_index]
                    elif current_response == 'focuscameraonspecificyoloobject':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        camera_vertical_pos = 'forward'
                        yolo_look = True
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        look_object = movement_response.split('~~')[1]
                    elif current_response == 'followuser':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        camera_vertical_pos = 'up'
                        follow_user = True
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    elif current_response == 'findunseenyoloobject':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        camera_vertical_pos = 'forward'
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        yolo_find = True
                        scan360 = 0
                        nav_object = movement_response.split('~~')[1]
                        rando_list = [1,2]
                        rando_index = random.randrange(len(rando_list))
                        rando_num = rando_list[rando_index]
                    else:
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        print('failed response')
                        
                    
                    
                    time.sleep(0.1)
            except:
                print(traceback.format_exc())
                
            
            
        except:
            print(traceback.format_exc())
yolo_nav = False
yolo_find = False
yolo_look = False
follow_user = False
if __name__ == "__main__":
    try:
        be_still = True
        last_time_seen = time.time()
        transcribe_thread = threading.Thread(target=listen_and_transcribe)  # Adding the transcription thread
        transcribe_thread.start() 
        camera = Picamera2()
        camera_config = camera.create_still_configuration()
        camera.configure(camera_config)
        camera.start()
        time.sleep(1)

        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["2"], arduino_address)
                        
        print('waiting to be called')
        ina219 = INA219(addr=0x42)
        while True:
            try:

                distance = int(read_distance_from_arduino())
                with open('current_distance.txt','w+') as f:
                    f.write(str(distance))
                break
            except:
                print(traceback.format_exc())
                
                continue
        while True:
            time.sleep(0.25)
            current = ina219.getCurrent_mA()
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            per = max(0, min(per, 100))  # Clamp percentage to 0-100 range
            per = (per * 2) - 100
            print('\nBattery Percent: ')
            print(per)
            with open('batt_per.txt', 'w+') as file:
                file.write(str(per))
            chat_history = []
            
            # Capture and resize the image, returning it as an array
            frame = capture_image(camera)
            cv2.imwrite('this_temp.jpg', frame)  # Save if needed for reference
            with open('last_phrase3.txt','r') as file:
                last_phrase = file.read()
            with open('last_phrase3.txt', 'w+') as file:
                file.write('')
            try:
                the_index_now = last_phrase.split(' ').index('echo')
                name_heard = True
            except ValueError:
                name_heard = False
            try:
                the_index_now = last_phrase.split(' ').index('robot')
                name_heard = True
            except ValueError:
                pass
            if name_heard:
                print("Name heard, initializing...")
                

                
              
                
          
                print('starting thread')
                movement_loop(camera)
                
            time.sleep(0.025)

    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
        
    finally:
        camera.close()
