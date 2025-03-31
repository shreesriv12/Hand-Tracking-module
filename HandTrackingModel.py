#added comments for better understanding
# press q to exit 


import cv2 # OpenCV for image processing
import mediapipe as mp # Google's library for hand tracking
import time #Used for calculating FPS (frames per second)
import numpy as np # Ensures proper handling of images


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): # False means it runs in video mode (detects hands in real-time),Detects up to 2 hands,Minimum detection confidence (default 50%),Minimum tracking confidence.
        self.mode = mode 
        self.maxHands = maxHands 
        self.detectionCon = detectionCon 
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #Calls the MediaPipe Hands module.
        self.hands = self.mpHands.Hands(  # Initializes the hand tracking model.
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils  #mpDraw → Used to draw hand landmarks on the screen.

    def findHands(self, img, draw=True): #Converts the BGR (default OpenCV format) to RGB for MediaPipe processing.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = np.ascontiguousarray(imgRGB) # Ensures efficient memory storage.

        # Process the image
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True): #Creates an empty list lmList to store landmark positions.
        lmList = []
        if self.results.multi_hand_landmarks: #Checks if hands are detected.
            if handNo < len(self.results.multi_hand_landmarks): #If multiple hands exist, selects a specific hand using handNo.
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape  #Extracts each landmark ID and its (x, y) coordinates.
                    cx, cy = int(lm.x * w), int(lm.y * h) #Converts the normalized lm.x and lm.y to pixel coordinates.
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  #If draw=True, draws circles at each landmark.
        return lmList


def main():
    pTime = 0
    cTime = 0 #pTime & cTime are used to calculate FPS.
    cap = cv2.VideoCapture(0) #Initializes video capture using webcam (0 = default camera).


    if not cap.isOpened():
        print("Error: Could not open camera. Trying alternative...")
        cap = cv2.VideoCapture(1)  # Try alternative camera(1)
        if not cap.isOpened():
            print("Error: No camera found.")
            return

    detector = handDetector()

    while True:  #Creates an instance of handDetector.
        success, img = cap.read()
        if not success:
            print("Failed to grab frame") #Continuously reads frames from the camera.
            break

        img = detector.findHands(img) #Calls findHands() to detect hands.
        lmList = detector.findPosition(img) #Calls findPosition() to get hand landmarks.
        if len(lmList) != 0:
            print(lmList[4]) #If landmarks exist, prints coordinates of landmark 4 (tip of the thumb).

        cTime = time.time()
        fps = 1 / (cTime - pTime) #Calculates Frames Per Second (FPS).
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3) #Displays FPS on the screen.

        cv2.imshow("Image", img) #Shows the video feed in a window.
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()  #Releases the camera and closes all OpenCV windows when the loop exits.


if __name__ == "__main__":
    main()