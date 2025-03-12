from fastapi import FastAPI, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn import svm
import uvicorn
import pickle
from models.model import train_model, train_and_evaluate
from utils.preprocessing import preprocess_data
from utils.database import connect_to_mongo, insert_record, get_predictions
import os
templates = Jinja2Templates(directory="templates")

model = None
scaler = None


def load_model():
    global model
    global scaler
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
    except FileNotFoundError:
        model, scaler = train_and_evaluate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    pass


app = FastAPI(lifespan=lifespan)


class PatientData(BaseModel):
    """
    age
    sex
    chest pain type (4 values)
    resting blood pressure
    serum cholestoral in mg/dl
    fasting blood sugar > 120 mg/dl
    resting electrocardiographic results (values 0,1,2)
    maximum heart rate achieved
    exercise induced angina
    oldpeak = ST depression induced by exercise relative to rest
    the slope of the peak exercise ST segment
    number of major vessels (0-3) colored by flourosopy
    thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
    """

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictResponse(BaseModel):
    prediction: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictResponse)
def predict_heart_disease_v2(data: PatientData):

    collection = connect_to_mongo()
    record = data.model_dump()
    X = preprocess_data(record)
    if model is not None and scaler is not None:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        print("Prediction: ", prediction)
    else:
        return {"error": "Model not yet trained"}
    record["prediction"] = int(prediction)
    insert_record(collection, record)
    return {"prediction": int(prediction)}

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    collection = connect_to_mongo()
    data = get_predictions(collection)
    for item in data:
        item["_id"] = str(item["_id"])
    return templates.TemplateResponse("history.html", {"request": request, "predictions": data})

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
