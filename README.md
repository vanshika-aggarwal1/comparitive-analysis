# Comparative Analysis of different CPU Utilization Forecasting Models

This repository contains the code, preprocessing pipeline, and results for a comparative study of four forecasting approaches — **Persistence (Naive)**, **ARIMA**, **XGBoost**, and **LSTM** — applied to predicting cloud CPU utilization using real-world workload data from the **Google 2019 Cluster Trace**.

The goal of this project is to evaluate how well simple statistical baselines compare against machine learning and deep learning models for forecasting cloud resource demand, with the broader motivation of supporting **green cloud computing** — accurate demand prediction enables proactive resource scaling, reducing over-provisioning and unnecessary energy consumption in data centers.

---

## Dataset

This project uses the **Google 2019 Cluster Trace** sample, distributed via Kaggle:

🔗 https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

The raw dataset is **not included in this repository** due to its size. To reproduce the results:

1. Download the dataset from the Kaggle link above (a free Kaggle account is required).
2. Place the downloaded CSV file in the main folder.
3. **Rename the downloaded file** as bord_traces_data.csv — the file Kaggle provides may download with a different name, so this step is necessary for the notebooks to locate it correctly.
4. Run `preprocessing.ipynb` from start to finish. This will generate the cleaned, model-specific datasets inside `data/`:
   - `cleaned_cpu_utilization.csv`
   - `persistence_data.csv`
   - `arima_data.csv`
   - `xgboost_data.csv`
   - `lstm_data.csv`

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and prepare the dataset

Follow the steps in the [Dataset](#dataset) section above before running any notebooks.

### 5. Run the notebooks in order

Launch Jupyter and run each notebook sequentially:

```bash
jupyter notebook
```

1. `preprocessing.ipynb` — must be run first; produces all cleaned datasets used by the models.
2. `persistence.ipynb`
3. `arima.ipynb`
4. `xgboost.ipynb`
5. `lstm.ipynb`

Each model notebook saves its evaluation metrics and prediction plots to the `results/` folder.

