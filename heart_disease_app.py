from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

# Loading ML model
model = pickle.load(open("model.pkl", "rb"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Bind home function to URL
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "Heart Disease Classifier.html", {"request": request}
    )


# Bind predict function to URL
@app.post("/predict")
async def predict(request: Request):
    # Retrieve form data
    form_data = await request.form()

    # Extract features from form data
    features = [float(value) for value in form_data.values()]

    # Convert features to array
    array_features = np.array(features).reshape(1, -1)

    # Make predictions
    prediction = model.predict(array_features)

    # Check the output values and retrieve the result with an HTML tag based on the value
    if prediction[0] == 1:
        result = "The patient is not likely to have heart disease!"
    else:
        result = "The patient is likely to have heart disease!"

    return templates.TemplateResponse(
        "Heart Disease Classifier.html", {"result": result, "output": prediction[0]}
    )
