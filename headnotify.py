
from imutils import face_utils
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import time
import numpy as np
import pywintypes
import win32api
from win10toast import ToastNotifier


# print the notificaion about on which mode code(eyeX) works(for osx)
def notify(title, text):
    toaster = ToastNotifier()
    toaster.show_toast(title ,text , duration=3)    


def direction(nose_point, anchor_point, r, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * r:
        return 'right'
    elif nx < x - multiple * r:
        return 'left'

    if ny > y + multiple * r:
        return 'down'
    elif ny < y - multiple * r:
        return 'up'

    return 'focus'


# all counters are initialised
input_run = 0

# initially all mode's are deactivated
INPUT_MODE = True


# initialised color variable
LIGHT_BLUE = (176,196,222)
LAVENDER_COLOR = (230,230,250)
LINEN_COLOR = (250,240,230)
THISTLE_COLOR = (216,191,216)


# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]



# Video capture
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

while True:
    # Grab the frame from the threaded video file stream,
    # resize it, and convert it to grayscale channels
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        #selecting first face
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then convert
    # the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    


    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    nose = shape[nStart:nEnd]
    

    #initialize the nose point
    nose_point = (nose[3, 0], nose[3, 1])

    
    
   
    if INPUT_MODE:
        if input_run == 0:
            notify("Input mode ON" ,"You are observed by webcam dont look away")
            input_run = 1

        # finding mid point of the screen to draw circle and line
        x = int(cam_w/2)
        y = int(cam_h/2)
        ANCHOR_POINT = (x,y)
        r = 50
        cv2.circle(frame, (x,y), r, LIGHT_BLUE, 4)
        cv2.line(frame, ANCHOR_POINT, nose_point, LAVENDER_COLOR, 4)


        # drag the mouse according to head movements
        dir = direction(nose_point, ANCHOR_POINT, r)
        print("Direction is",dir)
        if dir == 'right':
            time.sleep(10)
            if dir == 'right':
                notify("Don't look right","you were looking right for 5 sec")
        elif dir == 'left':
            time.sleep(10)
            if dir == 'left':
                notify("Don't look Left","you were looking Left for 5 sec")
       
    

    # to display the frame from webcam
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # to exit from running, uses "esc" key
    if key == 27:
        break

# remove all the widows which are created
cv2.destroyAllWindows()
vid.release()


