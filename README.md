-----

# 🧠 Machine Learning – Customer Purchase Prediction

This project predicts the probability of a customer making a purchase on an e-commerce site based on their browsing data. It includes:

  - A **FastAPI API** for model deployment.
  - A **Streamlit application** for interactive use (file upload or manual input).

-----

## Directory Structure

  - **`online_shoppers_intention.csv`**: The original dataset used for training and analysis.
  - **`modele_random_forest.joblib`**: The trained and saved Random Forest model (with the exact expected column structure).
  - **`preprocessing.py`**: A utility script to encode input data exactly as it was during model training (applies `get_dummies` to `Month` and `VisitorType`, handles the `Weekend` boolean, and dynamically adds/removes columns).
  - **`api.py`**: A FastAPI API with two endpoints:
      - `/predict_file/`: Upload a CSV file to get predictions.
      - `/predict_manual/`: Use manual JSON input for instant predictions.
  - **`app_streamlit.py`**: An interactive Streamlit application with two modes:
      - Upload a CSV file for batch predictions.
      - Manually enter variables for an instant prediction (with definitions and optimized default values).
  - **`requirements.txt`**: A list of the necessary Python dependencies for the project.

-----

## Installation

1.  **Clone the repository and navigate into the directory.**
2.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage

### 1\. Launch the FastAPI API

```bash
uvicorn api:app --reload
```

  - Access the interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  - There are two main endpoints:
      - **/predict\_file/**: Upload a CSV with the same columns as the original dataset.
      - **/predict\_manual/**: Send a JSON object (see the example in the Swagger docs).

### 2\. Launch the Streamlit Application

```bash
streamlit run app_streamlit.py
```

  - Access it in your browser (usually at http://localhost:8501).
  - It has two modes:
      - **Predict from File**: Upload a CSV, then view and download the results.
      - **Manual Input**: Use an interactive form with descriptions and optimized default values to predict a purchase.

-----

## File Details

### online\_shoppers\_intention.csv

  - The original dataset (12,330 browsing sessions, 18 columns).
  - Used for training and validating the model.

### modele\_random\_forest.joblib

  - A Random Forest model trained on the encoded data (with `get_dummies` applied to `Month` and `VisitorType`, and `Weekend` as a boolean).
  - Do not modify this file to ensure prediction compatibility.

### preprocessing.py

  - A utility function `preprocess_input` that:
      - Applies `get_dummies` to `Month` and `VisitorType` (`drop_first=True`).
      - Leaves `Weekend` as a boolean.
      - Adds any missing columns expected by the model.
      - Removes any unexpected columns.
      - Reorganizes the columns into the exact order the model expects.
  - This ensures compatibility between the input data and the model.

### api.py

  - A FastAPI API with two endpoints:
      - `/predict_file/`: Upload a CSV and get back predictions and probabilities.
      - `/predict_manual/`: Use manual JSON input and get back a prediction and probability.
  - It uses the preprocessing script to ensure data compatibility.

### app\_streamlit.py

  - An interactive Streamlit web application.
  - It has two modes:
      - **Predict from File**: Upload a CSV, display the results, and allow downloading.
      - **Manual Input**: A form with helpful definitions and default values optimized to maximize the probability of a purchase.
  - It uses the preprocessing script to ensure data compatibility.

### requirements.txt

  - A list of the required Python dependencies:
      - `fastapi`
      - `uvicorn`
      - `streamlit`
      - `pandas`
      - `scikit-learn`
      - `joblib`

-----

## Tips

  - **Always use the provided preprocessing script** for any new data you want to predict.
  - **Do not modify the model's structure** without retraining and saving a new, compatible model.
  - **If you encounter feature errors** (unexpected or missing columns), make sure the preprocessing step is being applied correctly.

-----

## Authors & Contact

  - An ML educational project.
  - For any questions or improvements, please open an issue or contact the repository owner.
