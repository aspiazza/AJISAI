from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model
import uvicorn
from icecream import ic

app = FastAPI()
templates = Jinja2Templates(directory="Templates/")

model_dir = 'F:\\Saved-Models\\Dog-Cat-Models\\First_Generation_dog_cat_optuna.h5'
model = load_model(model_dir)


@app.get('/')
def index():
    return 'You done fucked up bitch'


@app.get("/form")
def form_post(request: Request):
    result = "Type a number"
    return templates.TemplateResponse('index.html', context={'request': request, 'result': result})


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
