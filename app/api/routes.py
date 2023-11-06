from fastapi import APIRouter,  UploadFile, File, Form
import os
import cv2
import face_recognition
from starlette.responses import JSONResponse
import uuid
from starlette.responses import JSONResponse
from fer import FER
import dlib
import mediapipe as mp
import math
from imutils import face_utils
from scipy.spatial import distance


router = APIRouter()


@router.post("/api/v1/id_verification")
async def verify_id_and_face(input_video: UploadFile = File(...), id_card_image: UploadFile = File(...)):
    try:
        folder_name = "videos"
        video_file_name = input_video.filename
        id_card_name = id_card_image.filename

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        video_file_path = os.path.join(folder_name, video_file_name)
        id_card_path = os.path.join(folder_name, id_card_name)

        with open(video_file_path, "wb") as video_file:
            video_file.write(input_video.file.read())

        with open(id_card_path, "wb") as id_card_file:
            id_card_file.write(id_card_image.file.read())

        face_image = face_recognition.load_image_file(id_card_path)
        face_image_encoding = face_recognition.face_encodings(face_image)[0]

        video_capture = cv2.VideoCapture(video_file_path)
        frame_count = 0
        face_count = 0

        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                if frame_count == 0:
                    if os.path.exists(video_file_path):
                        os.remove(video_file_path)

                    return JSONResponse(content={"message": "Video not detected!! try again"})

                if face_count < 2:
                    if os.path.exists(video_file_path):
                        os.remove(video_file_path)
                    return JSONResponse(content={"message": "User/ID face missing! try again"})

            frame_count += 1

            if frame_count % 15 == 0:
                face_locations = face_recognition.face_locations(frame)
                face_count = len(face_locations)

                if face_count == 2:
                    user_face_location, held_face_location = face_locations
                    user_face_encoding = face_recognition.face_encodings(frame, [user_face_location])[0]
                    held_face_encoding = face_recognition.face_encodings(frame, [held_face_location])[0]
                    
                    face_distance1 = face_recognition.face_distance([face_image_encoding], held_face_encoding)
                    face_distance2 = face_recognition.face_distance([user_face_encoding], held_face_encoding)

                    threshold = 0.6
                    
                    if face_distance1[0] < threshold:
                        if face_distance2[0] < threshold:
                            if os.path.exists(video_file_path):
                                os.remove(video_file_path)
                            if os.path.exists(id_card_path):
                                os.remove(id_card_path)
                            return JSONResponse(content={"message": "Face matching successful!"})
                        else:
                            if os.path.exists(video_file_path):
                                os.remove(video_file_path)
                            if os.path.exists(id_card_path):
                                os.remove(id_card_path)
                            return JSONResponse(content={"message": "Face matching failed!"})
                            
                    else:
                        if os.path.exists(video_file_path):
                            os.remove(video_file_path)
                        if os.path.exists(id_card_path):
                            os.remove(id_card_path)
                        return JSONResponse(content={"message": "Face matching failed!"})
                
                if face_count > 2:
                    if os.path.exists(video_file_path):
                        os.remove(video_file_path)
                    if os.path.exists(id_card_path):
                        os.remove(id_card_path)
                    return JSONResponse(content={"message": "More than 2 faces detected"})

    except Exception as e:
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        if os.path.exists(id_card_path):
            os.remove(id_card_path)
        return JSONResponse(content={"message": f"There was an error processing the files: {e}"})



def analyze_hand_gestures(frame):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_hands = []

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            detected_hands.append(landmarks)

    for landmarks in detected_hands:
        hand_landmarks = landmarks.landmark

        wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        distance = math.dist((wrist.x, wrist.y), (middle_finger_tip.x, middle_finger_tip.y))
        if distance > 0.2:
            return True 

    return False


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def detect_blinks(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2

        if ear < 0.3:
            return True

    return False

@router.post("/api/v1/analyze_video")
async def analyze_video(input_video: UploadFile = File(...), emotions: str = Form(...)):
    try:
        folder_name = "videos"
        video_filename = str(uuid.uuid4()) + ".mp4"
        video_path = os.path.join(folder_name, video_filename)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(video_path, "wb") as video_file:
            video_file.write(input_video.file.read())

        emotions_to_detect =  emotions.split(',')

        emotion_model = FER()

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detected_emotions = []

        detector = dlib.get_frontal_face_detector()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(current_dir, "data_models", "shape_predictor_68_face_landmarks.dat")
        face_cascade_path = os.path.join(current_dir, "data_models", "haarcascade_frontalface_default.xml")

        predictor = dlib.shape_predictor(predictor_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            if frame_count % 8 == 0:

                emotions = emotion_model.detect_emotions(frame)

                if emotions:
                    dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                    detected_emotions.append(dominant_emotion)

                face_cascade = cv2.CascadeClassifier(face_cascade_path)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_center_x = x + w // 2

                    relative_x = face_center_x / frame.shape[1]

                    if relative_x < 0.4:
                        detected_emotions.append('turn head to the left')
                    elif relative_x > 0.6:
                        detected_emotions.append('turn head to the right')

                if 'wave' not in detected_emotions:
                    hand_wave_detected = analyze_hand_gestures(frame)

                    if hand_wave_detected:
                        detected_emotions.append('wave')

                if 'blink' not in detected_emotions:
                    blink_detected = detect_blinks(frame, detector, predictor)
                    
                    if blink_detected:
                        detected_emotions.append('blink')


        cap.release()


        
        response = [emotion for emotion in list(set(detected_emotions)) if emotion in emotions_to_detect]

        os.remove(video_path)

        return JSONResponse(content={"emotions": response})

    except Exception as err:
        os.remove(video_path)
        return JSONResponse(content={"message": str(err)})