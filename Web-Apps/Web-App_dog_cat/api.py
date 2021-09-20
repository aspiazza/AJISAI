from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, Request, Form
from tensorflow.keras import preprocessing
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from keras.models import load_model
from typing import List
from PIL import Image
import io
import numpy as np
import uvicorn
from icecream import ic

app = FastAPI()
app.mount("/Templates", StaticFiles(directory="Templates"), name="Templates")

model_dir = 'F:\\Saved-Models\\Dog-Cat-Models\\First_Generation_dog_cat_optuna.h5'
model = load_model(model_dir)


@app.get('/')
async def index():
    return RedirectResponse(url="/Templates/index.html")


@app.post('/prediction_page')
async def prediction_form(dogcat_img: UploadFile = File(...)):
    dogcat_img_bytes = dogcat_img.file.read()

    pp_dogcat_image = preprocessing.image.load_img(dogcat_img_bytes, target_size=(150, 150))
    pp_dogcat_image_arr = preprocessing.image.img_to_array(pp_dogcat_image)
    input_arr = np.array([pp_dogcat_image_arr])
    prediction = np.argmax(model.predict(input_arr), axis=-1)

    print(prediction)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

'''
@app.post('/prediction_page')
async def prediction_form(dogcat_img: bytes = Form(...)):
    ic(dogcat_img)
    ic(type(dogcat_img))


@app.post('/prediction_page')
async def prediction_form(dogcat_img: bytes = Form(...)):
    dogcat_image_arr = np.array(Image.open(BytesIO(dogcat_img)))
    image = preprocessing.image.load_img(dogcat_img, target_size=(150, 150))
    input_arr = preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = np.argmax(model.predict(input_arr), axis=-1)
    print(prediction)
'''
