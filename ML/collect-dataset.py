import csv 
import cv2
import mediapipe as mp

# import the drawing utils and hand recognition module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cols = [
    'p1_x',
    'p1_y',
    'p2_x',
    'p2_y',
    'p3_x',
    'p3_y',
    'p4_x',
    'p4_y',
    'p5_x',
    'p5_y',
    'p6_x',
    'p6_y',
    'p7_x',
    'p7_y',
    'p8_x',
    'p8_y',
    'p9_x',
    'p9_y',
    'p10_x',
    'p10_y',
    'p11_x',
    'p11_y',
    'p12_x',
    'p12_y',
    'p13_x',
    'p13_y',
    'p14_x',
    'p14_y',
    'p15_x',
    'p15_y',
    'p16_x',
    'p16_y',
    'p17_x',
    'p17_y',
    'p18_x',
    'p18_y',
    'p19_x',
    'p19_y',
    'p20_x',
    'p20_y',
    'p21_x',
    'p21_y',
    'sign'
]

rows = []

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        k = cv2.waitKey(2)
        if k & k == ord('q'):
            break
        elif k == ord('t'):
            mdict = {}
            for i, lnm in enumerate(hand_landmarks.landmark):
                mdict['p'+str(i+1)+"_x"] = lnm.x
                mdict['p'+str(i+1)+"_y"] = lnm.y
            mdict['sign'] = 'G'
            rows.append(mdict)
            print(rows)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    

# name of csv file 
filename = "dataset.csv"
    
# writing to csv file 
with open(filename, 'a') as csvfile: 
    # creating a csv dict writer object 
    writer = csv.DictWriter(csvfile, fieldnames = cols) 
        
    # writing headers (field names) 
    # writer.writeheader() 
        
    # writing data rows 
    writer.writerows(rows) 

cap.release()
