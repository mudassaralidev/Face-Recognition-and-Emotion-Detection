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


## Start the server locally without using docker

Run the following command to start the server at port: 8000

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```