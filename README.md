\# 🌦️ LankaWeather – Sri Lankan Multi-District Weather Forecasting and Climate Intelligence System



An intelligent weather forecasting platform developed using \*\*Ensemble Machine Learning\*\* to predict next-day weather conditions in Sri Lanka.



\## 📌 Project Overview



LankaWeather is a machine learning-based weather intelligence system designed to analyze historical weather data and predict next-day weather conditions.



The project applies multiple ensemble learning techniques to improve prediction accuracy and compare the performance of different machine learning approaches.



The system also provides an interactive Streamlit dashboard for weather prediction, model comparison, historical analysis, and feature importance visualization.



\## 🎯 Project Objectives



\- Analyze historical weather data in Sri Lanka.

\- Develop and compare multiple ensemble learning models.

\- Predict next-day maximum temperature, minimum temperature, and rainfall.

\- Evaluate model performance using standard regression metrics.

\- Visualize weather patterns and model results through an interactive dashboard.



\## 📊 Dataset



\- \*\*Country:\*\* Sri Lanka

\- \*\*Data Type:\*\* Daily historical weather data

\- \*\*Coverage:\*\* 2020–2026

\- \*\*Prediction Type:\*\* Next-day weather forecasting



\## 🤖 Machine Learning Models



Six machine learning approaches were developed and evaluated:



1\. Random Forest (Bagging)

2\. XGBoost (Boosting)

3\. LightGBM (Boosting)

4\. CatBoost (Boosting)

5\. Stacking Ensemble

6\. Hybrid Random Forest + ARIMA



\## 🌐 Dashboard Features



The interactive Streamlit dashboard includes:



\- \*\*Overview:\*\* Weather intelligence platform overview.

\- \*\*Model Comparison:\*\* Compare the performance of trained models.

\- \*\*Weather Prediction:\*\* Generate next-day weather predictions.

\- \*\*Feature Importance:\*\* Understand the contribution of input features.

\- \*\*Dataset Details:\*\* Explore dataset information.



\## 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- LightGBM

\- CatBoost

\- Statsmodels

\- Streamlit

\- Matplotlib

\- Seaborn

\- Joblib

\- Jupyter Notebook

\- Git \& GitHub

\- Git LFS



\## 📁 Project Structure



```text

LankaWeather\_Ensemble\_Project/

│

├── data/

│   └── raw/

│       └── sri\_lanka\_weather\_daily.csv

│

├── dashboard/

│   └── app.py

│

├── notebooks/

│   └── 01\_dataset\_investigation.ipynb

│

├── models/

│   ├── random\_forest.joblib

│   ├── xgboost.joblib

│   ├── lightgbm.joblib

│   ├── catboost.joblib

│   ├── stacking.joblib

│   ├── hybrid\_rf\_arima.joblib

│   └── project\_config.joblib

│

├── reports/

│   ├── actual\_vs\_predicted\_next\_day\_rainfall.png

│   ├── actual\_vs\_predicted\_next\_day\_temp\_max.png

│   ├── actual\_vs\_predicted\_next\_day\_temp\_min.png

│   ├── random\_forest\_feature\_importance.png

│   ├── test\_mae\_comparison.png

│   ├── test\_r2\_comparison.png

│   └── test\_rmse\_comparison.png

│

├── .gitattributes

├── .gitignore

└── README.md

```



\## ⚙️ Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/Senuri-Balasuriya/LankaWeather-Ensemble-Learning.git

```



\### 2. Navigate to the Project Directory



```bash

cd LankaWeather-Ensemble-Learning

```



\### 3. Create a Virtual Environment



```bash

python -m venv venv

```



\### 4. Activate the Virtual Environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 5. Install Dependencies



```bash

pip install pandas numpy scikit-learn xgboost lightgbm catboost statsmodels streamlit matplotlib seaborn joblib jupyter

```



\### 6. Run the Dashboard



```bash

streamlit run dashboard/app.py

```



The dashboard will open in your web browser.



\## 📈 Model Evaluation



The developed models are evaluated using:



\- Mean Absolute Error (MAE)

\- Mean Squared Error (MSE)

\- Root

## AI Tools Usage Acknowledgement

Generative AI tools, including ChatGPT and Google Gemini, were used during the development of this project to assist with understanding machine learning concepts, debugging code, improving code structure, and developing the Streamlit dashboard interface.

All model development, training, evaluation, result interpretation, and final project decisions were reviewed and verified by the project developer. The developer takes responsibility for the submitted implementation and results.