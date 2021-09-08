from fastapi import FastAPI, Request
from keras.models import load_model
import uvicorn

app = FastAPI()
model_dir = 'F:\\Saved-Models\\Dog-Cat-Models\\First_Generation_dog_cat_optuna.h5'
model = load_model(model_dir)


@app.get('/')
def index():
    return {'Hello': 'World'}


@app.post('/predict')
def predict_image(data):
    pass


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

# https://towardsdatascience.com/tensorflow-model-deployment-using-fastapi-docker-4b398251af75

# https://www.youtube.com/watch?v=mkDxuRvKUL8&list=WL&index=6&t=1139s
