## Getting Started with GTM Automation Backend App

This project was created with Python FastAPI

## Setup the project

Run these commands:

1. Create virtual environment

```
python -m venv venv
```

2. Activate virtual environment

```
source venv/bin/activate
```

3. Install dependencies to the project

```
pip install -r requirements.txt
```

## Running the Server on Docker Image

This guide will walk you through the process of running a FastAPI server within a Docker container. The Docker container will encapsulate your FastAPI application, making it easy to deploy and manage.

1. #### Prerequisites

Before you start, make sure you have the following prerequisites installed on your system:

- Docker: You can download and install Docker from [Docker's official website](https://www.docker.com/get-started).

2. #### Step-by-Step Instructions

Follow these steps to run your FastAPI server in a Docker container:

- Build a Dockerfile Image

Open a terminal and navigate to the directory where your Dockerfile is located. Build the Docker image using the following command:

```
docker build -t detection-app .
```

Replace detection-app with a suitable name for your Docker image.

- Run the Docker Container
  Once the Docker image is built, you can run a container from the image by running one of the following commands(first one is preferred):

1. Without detaching to container:

```
docker run -p 8000:8000 detection-app
```

2. With detaching to container:

```
docker run -d -p 8000:8000 detection-app
```

This command will start a Docker container running your FastAPI app, and it will map port 8000 from the container to port 8000 on your host machine.


- List all running docker containers:

```
docker ps
```

- Stop the docker container locally by running following command:

```
docker stop CONTAINER_ID
```
## Start the server locally without using docker

Run the following command to start the server at port: 8000

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```


## API parametes 
These APIs are expecting parameters with form data from Front End
1. ### Identification with ID image
  - ENDPOINT Name:  ```api/v1/id_verification```
  To verify the ID image with the provided video, this api takes two parameters
      1. input_video: video_file
      2. id_card_image: image of ID
  Here is the ScreenShot
    <img width="555" alt="image" src="https://github.com/IdeaForge-Technologies/api-project-face_recognition/assets/54367877/cff96a3b-a2ad-40da-b80f-3844ff06abb2">


2. ### Detect emotions with provided emotions
  - ENDPOINT Name:  ```api/v1/analyze_video```
  To detect the emotions of provided video, this api takes two parameters
      1. input_video: video_file
      2. emotions: comma seprated emotions(e.g smile, happy, sad etc)
  Here is the ScreenShot:
    <img width="575" alt="image" src="https://github.com/IdeaForge-Technologies/api-project-face_recognition/assets/54367877/22541dfe-cbc0-4dc9-9e01-89d7d4d70203">


