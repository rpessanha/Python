import cv2
import mediapipe as mp
import time
import numpy as np
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
canvas= None
pTime = 0
cTime = 0
write=True
# Resize da janela

#define the screen resulation
#screen_res = 1280, 720
#scale_width = screen_res[0] / img.shape[1]
#scale_height = screen_res[1] / img.shape[0]
#scale = min(scale_width, scale_height)
#resized window width and height
#window_width = int(img.shape[1] * scale)
#window_height = int(img.shape[0] * scale)
#cv2.WINDOW_NORMAL makes the output window resizealbe
#cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
#resize the window according to the screen resolution
#cv2.resizeWindow('Resized Window', window_width, window_height)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if canvas is None:
            canvas = np.zeros_like(img)
    if results.multi_hand_landmarks:
        

        for handLms in results.multi_hand_landmarks:
            
            image_height, image_width, _ = imgRGB.shape
            print(f'{handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x*image_width}'
            f'{handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y*image_height}')
            x2,y2 = int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x*image_width), int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y*image_height)
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
            
            else:
            # desenhar linha no canvas
                if write==True and int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y*image_height)<int(handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y*image_height):
                    canvas = cv2.line(canvas, (x1,y1),(x2,y2), [0,0,255], 4,cv2.LINE_8 )
        
            # depois os novos pontos tornam-me os anterioe
            x1,y1= x2,y2
            #cv2.circle(img, (int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x), int(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y)), radius=2, color=(0, 0, 255), thickness=-1)
            #cv2.circle(img, (handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x, handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y), 5, [50, 120, 255], 5) 
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                #cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    else:
        x1,y1 =0,0
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1024, 720)
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    img = cv2.addWeighted(img,0.7,canvas,0.7,0)
    cv2.imshow("Image", cv2.flip(img, 1))
    key=cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('c'):
        if write==True:
            write=False
        else:
            write=True