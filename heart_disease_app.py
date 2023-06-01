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
async def home(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("Heart Disease Classifier.html", context)


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

    context = {"result": result, "output": prediction[0]}

    return templates.TemplateResponse(
        request,
        "Heart Disease Classifier.html",
        context,
    )
