from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from keras.models import load_model
from typing import List
import uvicorn
from icecream import ic

app = FastAPI()
app.mount("/Templates", StaticFiles(directory="Templates"), name="Templates")


# model_dir = 'F:\\Saved-Models\\Dog-Cat-Models\\First_Generation_dog_cat_optuna.h5'
# model = load_model(model_dir)


@app.get('/')
async def index():
    return RedirectResponse(url="/Templates/index.html")


@app.post('/prediction_page')
async def prediction_form(dogcat_img: bytes = Form(...)):
    print(dogcat_img)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
