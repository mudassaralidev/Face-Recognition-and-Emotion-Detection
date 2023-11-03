from fastapi import APIRouter,  UploadFile, File
import os
import cv2
import face_recognition
from starlette.responses import JSONResponse


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