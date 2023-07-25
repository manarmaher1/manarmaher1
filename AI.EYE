import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import cvzone
from time import sleep

########################### blink
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)
########################### left blink
LI = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
LIratio = []

########################### blink

########################### right blink
RI = [253, 252, 256, 254, 339, 257, 258, 286, 259, 260, 359, 463]
RIratio = []
########################### blink
# # # # # # # # # # mouse

MOUSE = [11, 72, 73, 302, 303, 16, 85, 180, 315, 404, 61, 291]
Mratio = []

# # # # # # # # # # mouse
##########################

##########################
wCam, hCam = 640, 360
frameX = 340
frameY = 200
smoothening = 7
#########################

plocX, plocY = 0, 0
clocX, clocY = 0, 0


pyautogui.FAILSAFE = False
wscreen, hscreen = pyautogui.size()


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

# Read logo and resize
logo = cv2.imread('image.png')
size = 100
logo = cv2.resize(logo, (size, size))
# Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

while cap.isOpened():
    # blink
    # Region of Image (ROI), where we want to insert logo


    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # blink
    success, img = cap.read()

    img = cv2.flip(img, 1)
    img, faces = detector.findFaceMesh(img, draw=False)
    roi = img [-size - 10:-10, -size - 10:-10]
    roi[np.where(mask)] = 0
    roi += logo

    results = face_mesh.process(img)

    img_h, img_w, img_c = img.shape
    face_3d = []
    face_2d = []
    # 473,467
    # blink

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                if idx == 473 or idx == 468 or idx == 168 or idx == 6 or idx == 291 or idx == 199:
                    if idx == 473:
                        IRIS_2d = (lm.x * img_w, lm.y * img_h)
                        IRIS_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])

                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 90
            y = angles[1] * 90
            z = angles[2] * 90

            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -1.5:
                text = "SCROLLing Down"
                pyautogui.scroll(-200)
            elif x > 0.6:
                text = "SCROLLing Up"
                pyautogui.scroll(200)
            else:
                text = "Forward"


            iris_3d_projection, jacobian = cv2.projectPoints(IRIS_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(IRIS_2d[0]), int(IRIS_2d[1]))
            x2 = int(IRIS_2d[0])
            y2 = int(IRIS_2d[1])

            p2 = (int(IRIS_2d[0] + y * 12), int(IRIS_2d[1] - x * 12))
            x1 = int(IRIS_2d[0] + y * 12)
            y1 = int(IRIS_2d[1] - x * 12)
            cv2.rectangle(img, (frameX, frameY), (wCam - 235, hCam - 100),
                         (0, 255, 0), 2)

            x3 = np.interp(x1, (frameX, wCam - 235), (0, wscreen))
            y3 = np.interp(y1, (frameY, hCam - 100), (0, hscreen))

            cv2.line(img, p1, p2, (0, 0, 255), 2)

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(clocX, clocY)

            plocX, plocY = clocX, clocY

            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(img, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    if faces:
        face = faces[0]
        ritop = face[257]
        ridown = face[253]
        rileft = face[463]
        riright = face[359]
        lenghtVer1, _ = detector.findDistance(ritop, ridown)
        lenghtHor1, _ = detector.findDistance(rileft, riright)

        ratio1 = int((lenghtVer1 / lenghtHor1) * 100)
        RIratio.append(ratio1)
        if len(RIratio) > 3:
            RIratio.pop(0)
        ratioAvg1 = sum(RIratio) / len(RIratio)

        if ratioAvg1 < 35:
            pyautogui.click(button='right')


        # # # # # # # # # # right
        litop = face[159]
        lidown = face[23]
        lileft = face[130]
        liright = face[243]
        lenghtVer, _ = detector.findDistance(litop, lidown)
        lenghtHor, _ = detector.findDistance(lileft, liright)
        ratio = int((lenghtVer / lenghtHor) * 100)
        LIratio.append(ratio)
        if len(LIratio) > 3:
            LIratio.pop(0)
        ratioAvg = sum(LIratio) / len(LIratio)

        if ratioAvg < 35:
            pyautogui.click(button='left')
     

        # # # # # # # # right
        # # # # # # # # # # mouse

        mtop = face[11]
        mdown = face[16]
        mleft= face[61]
        mright = face[291]
        mlenghtVer, _ = detector.findDistance(mtop, mdown)
        mlenghtHor, _ = detector.findDistance(mleft, mright)

        ratio2 = int((mlenghtVer / mlenghtHor) * 100)
        Mratio.append(ratio2)
        if len(Mratio) > 3:
            Mratio.pop(0)
        ratioAvg2 = sum(Mratio) / len(Mratio)

        if ratioAvg2 > 45:
             pyautogui.click(button='right')


    cv2.imshow('AI-EYE', img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
