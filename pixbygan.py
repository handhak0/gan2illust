'''모듈 불러오기'''
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


import uvicorn
import os

from pyngrok import ngrok
import nest_asyncio


import sys
sys.path.append("C:/Users/sonso/Desktop/Git/멀티캠퍼스/04.FinalProject/손학영/Face_segmentation")

# 챗봇 용
import json
from chatbot import ChatbotMessageSender

# 모델 모듈불러오기
import segmentation

from UGATIT2 import UGATIT
from PIL import Image
from utils import *
import numpy as np

modelW = UGATIT()
modelW.build_model()
modelW.load('./YOUR_DATASET_NAME_params_latest.pt')



'''프로그램 시작'''

app = FastAPI()

# src 연결
app.mount("/static", StaticFiles(directory="static"), name="static") # css 파일
templates = Jinja2Templates(directory = "templates") # html 파일

# root 연결
@app.get('/', response_class = HTMLResponse)
async def root(request : Request) :
    return templates.TemplateResponse("home.html", context={"request": request})


@app.get('/home', response_class = HTMLResponse)
async def home(request : Request) :
    return templates.TemplateResponse("home.html", context={"request": request})

@app.get('/home_img/{img_num}')
async def home_img(img_num : int) :
    homeImg = os.listdir("./static/output")
    img_src = "./static/output/" + homeImg[img_num]
    return FileResponse(img_src)


# @app.get('/home_imgshow/{img_num}', response_class = HTMLResponse)
# async def home_imgshow(img_num : int) :
#     img_src = "./static/output/" + homeImg[img_num]
#     return templates.TemplateResponse("homeImg.html", context={"request": request})



@app.get('/login', response_class = HTMLResponse)
async def login(request : Request) :
    return templates.TemplateResponse("login.html", context={"request": request})

@app.post('/login_check/', response_class= HTMLResponse)
async def login_check(request : Request):
    return templates.TemplateResponse("home.html", context={"request":request})


@app.get('/description', response_class = HTMLResponse)
async def description(request : Request) :
    return templates.TemplateResponse("description.html", context={"request": request})


@app.get('/write', response_class = HTMLResponse)
async def write(request : Request) :
    return templates.TemplateResponse("write.html", context={"request": request})


@app.post("/runmodel/", response_class = HTMLResponse)
async def runmodel(request : Request, files: UploadFile = File(...), style : str = Form(...)):
    global fnm
    fnm = files.filename

    if style == "human" :
        file_location = f"./static/input/{fnm}"
        with open(file_location, "wb+") as file_object:
            file_object.write(files.file.read())
        # face segmentation 실행
        segmentation.segmentation(fnm)
    elif style == "webtoon" :
        file_location = f"./dataset/YOUR_DATASET_NAME/testA/{fnm}"
        with open(file_location, "wb+") as file_object:
            file_object.write(files.file.read())

        modelW.dataload()

        real_A, _ = next(iter(modelW.testA_loader))
        real_A = real_A.to(modelW.device)

        fake_A2B, _, fake_A2B_heatmap = modelW.genA2B(real_A)

        image = tensor2numpy(denorm(fake_A2B[0]))

        image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((256, 256)).convert('RGB'))
        image = Image.fromarray(image)

        output_src = f"./static/output/{fnm}"
        image.save(output_src)

        os.remove(file_location)

    return templates.TemplateResponse("output.html", context={"request": request})



# @app.post("/runstyle", response_class = HTMLResponse)
# async def runstyle(request : Request, files : UploadFile = File(...)) :
#     fnm = files.filename
#     file_location = f"./dataset/YOUR_DATASET_NAME/testA/{fnm}"
#     with open(file_location, "wb+") as file_object:
#         file_object.write(files.file.read())
#
#     modelW.dataload()
#
#     real_A, _ = next(iter(modelW.testA_loader))
#     real_A = real_A.to(modelW.device)
#
#     fake_A2B, _, fake_A2B_heatmap = modelW.genA2B(real_A)
#
#     image = tensor2numpy(denorm(fake_A2B[0]))
#
#     image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((256, 256)).convert('RGB'))
#     image = Image.fromarray(image)
#
#     output_src = f"./static/output/{fnm}"
#     image.save(output_src)
#
#     os.remove(file_location)
#
#     return templates.TemplateResponse("output.html", context={"request": request})


@app.get('/output', response_class = HTMLResponse)
async def output(request : Request) :

    return templates.TemplateResponse("output.html", context={"request": request})

@app.get('/output_img')
async def output_img() :
    global output_src
    output_src = f"./static/output/{fnm}"
    return FileResponse(output_src)


@app.get('/download')
async def download() :
    # file_path = './static/output/20211119_101611_12.png'
    return FileResponse(path = output_src, filename='character.png')


@app.get('/chatbot', response_class = HTMLResponse)
async def output(request : Request) :

    return templates.TemplateResponse("chatbot.html", context={"request": request})


@app.post('/chatbot/answer/', response_class = HTMLResponse)
async def chatAnswer(request : Request, question: str = Form(...)):
    question = question
    res = ChatbotMessageSender(question).req_message_send()
    answer = json.loads(res.text)
    chatAnswer = answer['bubbles'][0]['data']

    return templates.TemplateResponse("chat.html", context={"request": request, "chatAnswer" : chatAnswer})


# ngrok_tunnel = ngrok.connect(8000)
# print ('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
# uvicorn.run(app, host='0.0.0.0', port=8000)


uvicorn.run(app, port=8000)


