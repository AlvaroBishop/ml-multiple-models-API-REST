import time
from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from scripts import s3

from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os

from transformers import AutoImageProcessor
from transformers import pipeline
import torch

import warnings

warnings.filterwarnings('ignore')

# Esto es necesario para el human pose classification
model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


########### Download ML Models ##########
force_download = False # False

def download_model(model_name, local_path = "ml-models/"):
    local_path = local_path + model_name
    if not os.path.isdir(local_path) or force_download:
        s3.download_dir(local_path, model_name)

download_model('tinybert-sentiment-analysis')
download_model('tinybert-disaster-tweet')
download_model('vit-human-pose-classification')

sentiment_model = pipeline('text-classification', model='ml-models/tinybert-sentiment-analysis', device = device)
tweets_model = pipeline('text-classification', model='ml-models/tinybert-disaster-tweet', device = device)
pose_model = pipeline('image-classification', model='ml-models/vit-human-pose-classification', device=device, image_processor=image_processor)

########### Download ML ENDS   ##########


@app.get("/")
def read_root():
    return "Hello!!!!!!!!!!!"

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput): # NLPDataInput uses pydantic class for data validation task
    # model_name: str
    # text: list[str]
    # labels: list[str]
    # scores: list[float]
    # prediction_time: int

    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start) * 1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    return NLPDataOutput(model_name="tinybert-sentiment-analysis",
                         text=data.text,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)


@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()    
    output = tweets_model(data.text)
    end = time.time()
    prediction_time = int((end-start) * 1000)    

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]   

    return NLPDataOutput(model_name='tinybert-disaster-tweet',
                         text=data.text,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    start = time.time()
    urls = [str(x) for x in data.url]
    output = pose_model(urls)
    end = time.time()
    prediction_time = int((end-start) * 1000)    

    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]   

    return ImageDataOutput(model_name='vit-human-pose-classification',
                         url=data.url,
                         labels=labels,
                         scores=scores,
                         prediction_time=prediction_time)








if __name__=="__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)