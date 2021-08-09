import cv2
import numpy as np
import datetime
import os
import HandTrackingModule as htm

#################################
brushThickness = 10
eraserThickness = 200
#################################

now = datetime.datetime.now()
nowTime = now.strftime('%Y_%m_%d_%H_%M')

folderPath = "Header"
videoPath = "Video"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[4]
drawColor = (0, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

#######################################################
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'{videoPath}/output_{nowTime}.mp4', fourcc, fps, (int(width), int(height)))
#######################################################

while True:
    #1. Import image
    success, img = cap.read()
    if not success:
        print("프레임을 수신할 수 없습니다.")
        break

    #2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)
        #tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4. If Selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("selection Mode")

            # Checking for the click
            if y1 < 125:
                if 100<x1<240:
                    header = overlayList[0]
                    drawColor = (245, 226, 186)

                elif 340<x1<460:
                    header = overlayList[1]
                    drawColor = (130, 186, 249)

                elif 580<x1<700:
                    header = overlayList[2]
                    drawColor = (185, 162, 243)

                elif 820<x1<940:
                    header = overlayList[3]
                    drawColor = (164, 201, 140)

                elif 1060 < x1 < 1180:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        #5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 5, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("image", img)
    out.write(img)

    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()