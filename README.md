# Heart Disease Classifier

This is a web application built using FastAPI to predict the likelihood of heart disease based on user input.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Python (3.11.0)
- FastAPI 0.95.2)
- NumPy (1.23.5)

### Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/aquacodee/Heart-_disease_classification.git

2. Install the dependencies:
```shell
pip install -r requirements.txt
```

3. Run the application:

```shell
uvicorn main:app --reload
```

4. Open your web browser and go to http://localhost:8000 to access the Heart Disease Classifier.

### Usage
1. Fill in the form with the patient's information.

2. Click the "Result" button to predict the likelihood of heart disease.

3. The prediction result will be displayed on the page.

### Acknowledgements
This application uses machine learning models trained on the Heart Disease dataset. The models were developed using scikit-learn and saved using Pickle.

### License

This project is licensed under the [MIT License](LICENSE).
