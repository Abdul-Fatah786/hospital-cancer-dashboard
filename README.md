# 🏥 City Hospital: Live ML Training Dashboard

A PyTorch neural network that classifies breast tumours as **benign or malignant** using 30 cell-nucleus measurements from the Breast Cancer Wisconsin (Diagnostic) dataset.

The project includes an interactive **Streamlit dashboard** where users can modify training configurations, monitor loss curves, compare experiments, and evaluate model performance. It also includes a **FastAPI service** for serving predictions from the trained model.

> **Educational project:** This application uses a publicly available dataset for machine learning experimentation. It is not a medical device and must never be used to diagnose real patients.

## 🚀 Live Demo

The application is deployed on Streamlit Community Cloud.

**Live Dashboard:**  
https://hospital-cancer-dashboard-cptkqwxpumwfujgyhh5sxz.streamlit.app/

Explore the dashboard to:
- Analyze the breast cancer dataset.
- Experiment with neural network training settings.
- Monitor training and validation loss curves.
- Compare different machine learning experiments.
- Visualize model performance and confusion matrices.
- Test predictions using sample patients or uploaded CSV files.

## 🎯 The Question This Project Answers

A model can achieve almost zero training loss and still perform poorly on unseen data.

**How can we detect and fix overfitting in neural networks?**

This project explores that question through six experiments comparing:

1. Baseline neural network.
2. Deliberately overfitting model.
3. Dropout and weight decay.
4. Early stopping.
5. Smaller neural network.
6. Training without feature scaling.

The dashboard evaluates each experiment and classifies its behaviour as:

- **Healthy:** The model generalizes reasonably well.
- **Overfitting:** The model performs significantly better on training data than validation data.
- **Underfitting:** The model fails to learn sufficiently from training data.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Neural network development and training |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Dataset loading, preprocessing and evaluation |
| Streamlit | Interactive ML dashboard |
| FastAPI | Model prediction API |
| Matplotlib | Data visualization and loss curves |
| Seaborn | Statistical visualizations |
| Pytest | Automated testing |
| Docker | Containerization |

## 📂 Project Structure

```text
hospital-cancer-dashboard/
│
├── data/
│   ├── README.md
│   └── data.csv
│
├── notebooks/
│   └── hospital_experiments.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── experiments.py
│   ├── evaluate.py
│   ├── train.py
│   ├── predict.py
│   ├── eda.py
│   └── make_data.py
│
├── models/
│   ├── hospital_model.pt
│   ├── scaler.pkl
│   └── best_config.json
│
├── reports/
│   ├── experiment_history.csv
│   ├── final_test_metrics.csv
│   └── figures/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── tests/
│
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
└── requests.http
```

## ⚙️ Setup in VS Code

### 1. Requirements

Install the following:

- Python 3.10 or higher.
- Visual Studio Code.
- Python extension for VS Code.
- Jupyter extension (optional).

### 2. Clone the repository

```bash
git clone <your-github-repository-url>
cd hospital-cancer-dashboard
```

### 3. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 5. Select Python interpreter

In VS Code:

1. Press `Ctrl + Shift + P`.
2. Search for `Python: Select Interpreter`.
3. Select the Python interpreter inside `.venv`.

### 6. Prepare the dataset

You can download the dataset from Kaggle and place `data.csv` inside the `data/` directory.

Alternatively, generate the dataset using scikit-learn:

```bash
python -m src.make_data
```

## ▶️ Running the Project

Always execute commands from the project root directory.

| Task | Command |
|---|---|
| Generate dataset | `python -m src.make_data` |
| Exploratory Data Analysis | `python -m src.eda` |
| Train the model | `python -m src.train` |
| Launch Streamlit | `streamlit run app.py` |
| Launch FastAPI | `uvicorn api.main:app --reload` |
| Run tests | `pytest` |

**Local application URLs:**

- Streamlit Dashboard: http://localhost:8501
- FastAPI Documentation: http://127.0.0.1:8000/docs

The Predict tab and API require a trained model. Run `python -m src.train` before using them.

## 📊 Dashboard Features

### 1. Data Explorer

- Dataset class distribution.
- Feature distributions by tumour diagnosis.
- Correlation between features and diagnosis.
- Feature scaling comparisons.

### 2. Training Lab

An interactive training environment where users can modify:

- Hidden neurons.
- Dropout rate.
- Learning rate.
- Weight decay.
- Early stopping patience.
- Maximum training epochs.
- Feature scaling.

After clicking **Train Model**, users can monitor:

- Training loss.
- Validation loss.
- Accuracy.
- Recall.
- Cancer cases detected and missed.
- False-positive predictions.
- Confusion matrix.
- Model diagnosis: Healthy, Overfitting or Underfitting.

Training history can also be downloaded as a CSV file.

### 3. Experiments

Compare six planned experiments through:

- Experiment comparison tables.
- Training and validation loss curves.
- Performance metric comparisons.
- Overfitting and underfitting analysis.

### 4. Predict

- Select a held-out test patient.
- Compare actual and predicted diagnoses.
- View prediction probabilities.
- Upload a CSV file containing the required 30 feature columns.

## 🔬 How the Model Avoids Data Leakage

The project follows a structured data-splitting and evaluation process:

1. The dataset is split first using stratified 60/20/20 splitting.
2. The scaler is fitted on the training dataset only.
3. Hyperparameters are selected using the validation dataset.
4. The selected configuration is retrained using training and validation data.
5. The scaler is refitted on the combined training and validation data.
6. The final test dataset is evaluated once at the end.

**Dataset distribution:**

| Dataset | Samples |
|---|---:|
| Training | 341 |
| Validation | 114 |
| Testing | 114 |
| Total | 569 |

The dashboard does not use the final test dataset for interactive model selection.

## 🧠 Model Diagnosis Rules

The dashboard uses the following rules to identify model behaviour:

| Classification | Condition |
|---|---|
| Underfitting | Final training loss > 0.30 |
| Overfitting | Final validation loss > 1.30 × best validation loss from an earlier epoch |
| Healthy | Neither condition is satisfied |

The last epoch is excluded when calculating the earlier best validation loss to avoid comparing the final epoch against itself.

## 📈 Final Model Results

The trained PyTorch neural network achieved strong classification performance on the held-out test dataset.

### Test Performance Metrics

| Metric | Result |
|---|---:|
| Test Loss | 0.11256 |
| Accuracy | **98.25%** |
| Precision | **100.00%** |
| Recall | **95.24%** |
| F1-Score | **97.56%** |

### Confusion Matrix

| Metric | Count |
|---|---:|
| True Positives (TP) | 40 |
| False Negatives (FN) | 2 |
| False Positives (FP) | 0 |
| True Negatives (TN) | 72 |

### Performance Analysis

- **98.25% Accuracy:** Correctly classified 112 out of 114 test samples.
- **100% Precision:** All 40 samples predicted as malignant were actually malignant.
- **95.24% Recall:** Successfully identified 40 out of 42 malignant cases.
- **97.56% F1-Score:** Achieved a strong balance between precision and recall.
- **Zero False Positives:** No benign samples were incorrectly classified as malignant.

The results demonstrate strong performance on this particular test split. However, the model missed two malignant cases. In medical classification, false negatives are particularly important because they represent malignant tumours incorrectly classified as benign.

These results do not establish clinical reliability or performance on external hospital populations.

## 📋 Experiment Analysis

Use `reports/experiment_history.csv` and `reports/figures/loss_curves.png` to investigate:

1. Which experiment has the biggest gap between training and validation loss?
2. Did dropout and weight decay reduce overfitting compared with experiment 2?
3. Did early stopping select a better epoch?
4. Did the smaller model generalize better than the baseline?
5. What happened when feature scaling was removed?
6. Which experiment detected the most malignant cases?

## 🔌 FastAPI Endpoints

The project includes a FastAPI service for making predictions using the trained model.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service and model status |
| GET | `/model-info` | View model configuration and test metrics |
| GET | `/features` | Retrieve the 30 feature names |
| GET | `/sample?malignant=true` | Retrieve a held-out example |
| POST | `/predict` | Predict one patient's tumour classification |
| POST | `/predict/batch` | Predict multiple patients |
| POST | `/predict/csv` | Upload a CSV for batch predictions |

### API Documentation

After starting the FastAPI server, visit:

http://127.0.0.1:8000/docs

The prediction endpoints support an optional classification threshold between 0 and 1.

## 🐳 Docker Deployment

Docker and Docker Compose configurations are included.

To build and start the services:

```bash
docker-compose up --build
```

The intended service ports are:

- Streamlit Dashboard: `8501`
- FastAPI: `8000`

The Docker configuration is included in the project; deployment through Docker should be tested separately.

## ☁️ Streamlit Cloud Deployment

The dashboard is deployed using Streamlit Community Cloud.

Deployment steps:

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select the `main` branch.
5. Set `app.py` as the main application file.
6. Deploy the application.

Ensure that required dependencies and application data are available in the repository or through the dashboard's data-upload functionality.

**Live Application:**  
https://hospital-cancer-dashboard-cptkqwxpumwfujgyhh5sxz.streamlit.app/

## ⚠️ Notes and Limitations

- The dataset contains only 569 samples, so the results may vary with different random splits and seeds.
- Experiment 5 uses 8 hidden neurons to make it different from the baseline.
- Experiment 6 removes feature scaling and may produce unstable validation loss.
- Recall is particularly important because false negatives represent missed malignant cases.
- A single test split is not sufficient to establish generalization to new hospitals or patient populations.
- The model is intended for educational machine learning experimentation only.
- It is not approved for clinical use and must not be used for medical diagnosis or treatment decisions.

## 👨‍💻 Author

**Abdul Fatah**

Computer Science Student | Software Developer | Machine Learning Enthusiast

GitHub: https://github.com/Abdul-Fatah786

LinkedIn: https://linkedin.com/in/abdulfatah7/

---

⭐ If you find this project useful, consider giving the repository a star!

**Built with Python, PyTorch, Streamlit and FastAPI.**