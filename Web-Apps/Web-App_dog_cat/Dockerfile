FROM python:3.8.12-slim-buster

COPY ./App /app/src
COPY ./requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host=localhost", "--reload"]
