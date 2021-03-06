# API for Dog Cat AI Container

from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile
from tensorflow_addons.activations import mish
from tensorflow.keras import preprocessing
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()
app.mount("/Templates", StaticFiles(directory="Templates/"), name="Templates")
templates = Jinja2Templates(directory="Templates")

# Change to whatever model you prefer so long as its compatible with preprocessing parameters
model_dir = 'F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat_optuna.h5'
model = load_model(model_dir)


# Prediction function
def predict_image(image):
    pp_dogcat_image = Image.open(image.file).resize((150, 150), Image.NEAREST).convert("RGB")
    pp_dogcat_image_arr = preprocessing.image.img_to_array(pp_dogcat_image)
    input_arr = np.array([pp_dogcat_image_arr])
    prediction = np.argmax(model.predict(input_arr), axis=-1)

    if str(prediction) == '[1]':
        answer = "It's a Dog"
    else:
        answer = "It's a Cat"
    return answer


# Index page
@app.get('/')
async def index():
    return RedirectResponse(url="/Templates/index.html")


# Prediction Page
@app.post('/prediction_page')
async def prediction_form(dogcat_img: UploadFile = File(...)):
    answer = predict_image(dogcat_img)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
