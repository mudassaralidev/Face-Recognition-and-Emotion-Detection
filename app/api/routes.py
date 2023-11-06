from fastapi import APIRouter, UploadFile, File, Form
import os
import uuid
from starlette.responses import JSONResponse
from app.helper_methods import  detect_emotions_and_head_pose, process_video, remove_file, write_file

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

        msg = process_video(video_file_path, id_card_path)

        return msg

    except Exception as e:
        remove_file(video_file_path)
        remove_file(id_card_path)
        return JSONResponse(content={"message": f"There was an error processing the files: {e}"})




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

        detected_emotions = detect_emotions_and_head_pose(video_path)
        
        response = [emotion for emotion in detected_emotions if emotion in emotions_to_detect]

        remove_file(video_path)

        return JSONResponse(content={"emotions": response})

    except Exception as err:
        remove_file(video_path)
        return JSONResponse(content={"message": str(err)})
