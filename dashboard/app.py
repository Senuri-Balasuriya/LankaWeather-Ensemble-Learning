
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LankaWeather | Climate Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "sri_lanka_weather_daily.csv"
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# 2. THEME AND CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        linear-gradient(120deg, rgba(5, 20, 16, 0.96), rgba(9, 35, 27, 0.91)),
        url("https://images.unsplash.com/photo-1448375240586-882707db888b");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #F0F7F2;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    display: none;
}

h1, h2, h3 {
    color: #F4FFF7 !important;
    font-family: 'Manrope', sans-serif !important;
}

p, label, span {
    color: inherit;
}

.brand {
    color: #A3E635;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 18px;
}

.main-title {
    color: #F4FFF7;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 12px;
}

.main-description {
    color: #C3D8CB;
    font-size: 16px;
    line-height: 1.7;
    max-width: 700px;
}

.header-line {
    border: none;
    height: 1px;
    background: rgba(190, 230, 205, 0.18);
    margin: 35px 0 25px 0;
}

/* Navigation */

div.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 12px;
    border: 1px solid rgba(190, 230, 205, 0.16);
    background: rgba(17, 48, 37, 0.75);
    color: #E6F3EA;
    font-weight: 600;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border-color: #A3E635;
    color: #D9FF8C;
    background: rgba(43, 82, 55, 0.85);
}

/* Panels */

.content-panel {
    background: rgba(13, 39, 30, 0.78);
    border: 1px solid rgba(190, 230, 205, 0.16);
    border-radius: 20px;
    padding: 25px;
    margin-top: 22px;
    backdrop-filter: blur(14px);
}

.section-label {
    color: #A3E635;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.section-title {
    color: #F4FFF7;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.section-description {
    color: #C3D8CB;
    font-size: 14px;
    margin-top: 8px;
    line-height: 1.6;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: rgba(18, 51, 38, 0.82);
    border: 1px solid rgba(190, 230, 205, 0.15);
    padding: 20px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    color: #BBD0C2 !important;
}

[data-testid="stMetricValue"] {
    color: #D9FF8C !important;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    color: #A3E635 !important;
}

/* Select boxes */

div[data-baseweb="select"] > div {
    background-color: #173B2D;
    border-color: rgba(190, 230, 205, 0.25);
    color: #FFFFFF;
}

.stSelectbox label, .stDateInput label {
    color: #DCECE1 !important;
    font-weight: 600;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border: 1px solid rgba(190, 230, 205, 0.15);
    border-radius: 12px;
}

/* Buttons */

.stDownloadButton button {
    background: #A3E635;
    color: #10251A;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

hr {
    border-color: rgba(190, 230, 205, 0.15);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PROJECT CONFIGURATION
# =========================================================

MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Hybrid RF + ARIMA",
        "Stacking Ensemble",
        "Random Forest"
    ],
    "MAE": [2.2180, 2.2272, 2.2638, 2.3052, 2.3286, 2.3367],
    "RMSE": [7.1961, 7.2079, 7.2263, 7.0949, 7.1928, 7.0916],
    "R²": [0.6892, 0.6880, 0.6713, 0.6927, 0.6624, 0.6848]
})

PREDICTION_MODELS = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Stacking Ensemble"
]

WEATHER_COLUMNS = [
    "temp_max_c",
    "temp_min_c",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "humidity_mean_pct"
]

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "CatBoost": "catboost.joblib",
    "Stacking Ensemble": "stacking.joblib"
}


# =========================================================
# 4. LOAD DATA AND MODELS
# =========================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    return df


def unwrap_model(obj):
    """
    Supports both a saved estimator and a dictionary
    containing a saved estimator.
    """

    if not isinstance(obj, dict):
        return obj

    preferred_keys = [
        "model",
        "pipeline",
        "best_model",
        "best_rf_model",
        "trained_model",
        "stacking_model",
        "stacking_pipeline"
    ]

    for key in preferred_keys:
        if key in obj and hasattr(obj[key], "predict"):
            return obj[key]

    for value in obj.values():
        if hasattr(value, "predict"):
            return value

    raise ValueError(
        "Could not find a prediction model in the saved file."
    )


@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        if not path.exists():
            continue

        try:
            saved_object = joblib.load(path)
            loaded_models[model_name] = unwrap_model(saved_object)
        except Exception as error:
            st.warning(
                f"Could not load {model_name}: {error}"
            )

    return loaded_models


@st.cache_resource
def load_project_config():
    path = MODEL_DIR / "project_config.joblib"

    if not path.exists():
        return {}

    return joblib.load(path)


# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

def create_features(df):
    """
    Generate historical weather features for next-day
    prediction. The final input columns are selected
    using the saved project configuration.
    """

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["city", "date"]).reset_index(drop=True)

    for column in WEATHER_COLUMNS:

        if column not in data.columns:
            continue

        for lag in [1, 2, 3, 7]:
            data[f"{column}_lag_{lag}"] = (
                data.groupby("city")[column].shift(lag)
            )

        data[f"{column}_rolling_3"] = (
            data.groupby("city")[column]
            .transform(
                lambda values: values.shift(1).rolling(3).mean()
            )
        )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * data["day_of_year"] / 365.25
    )
    data["day_of_year_cos"] = np.cos(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    # Keep these features because they are included
    # in the saved model configuration.
    data["day_sin"] = np.sin(
        2 * np.pi * data["day"] / 31
    )
    data["day_cos"] = np.cos(
        2 * np.pi * data["day"] / 31
    )

    return data


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def section_header(label, title, description):
    st.markdown(f"""
    <div class="content-panel">
        <div class="section-label">{label}</div>
        <div class="section-title">{title}</div>
        <div class="section-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def get_feature_columns(config):
    return config.get("feature_columns", [])


def get_target_columns(config):
    return config.get(
        "target_columns",
        [
            "next_day_temp_max",
            "next_day_temp_min",
            "next_day_rainfall"
        ]
    )


def get_prediction_row(df, city, feature_columns):
    city_data = df[df["city"] == city].copy()

    if city_data.empty:
        raise ValueError(f"No records found for {city}.")

    city_data = create_features(city_data)

    missing_features = [
        col for col in feature_columns
        if col not in city_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Some model features are missing: "
            + ", ".join(missing_features)
        )

    valid_rows = city_data.dropna(
        subset=feature_columns
    )

    if valid_rows.empty:
        raise ValueError(
            "Not enough historical data to generate features."
        )

    return valid_rows.iloc[-1]


def friendly_feature_name(name):
    labels = {
        "temp_max_c": "Maximum Temperature",
        "temp_min_c": "Minimum Temperature",
        "precipitation_mm": "Rainfall",
        "wind_speed_max_kmh": "Maximum Wind Speed",
        "humidity_mean_pct": "Average Humidity",
        "month": "Month",
        "day": "Day of Month",
        "year": "Year",
        "day_of_week": "Day of Week",
        "day_of_year": "Day of Year",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "city": "City"
    }

    if name in labels:
        return labels[name]

    for key, label in labels.items():
        if name.startswith(key + "_lag_"):
            lag = name.split("_lag_")[-1]
            return f"{label} - {lag} Day(s) Earlier"

        if name.startswith(key + "_rolling_"):
            return f"{label} - 3 Day Average"

    if name == "month_sin" or name == "month_cos":
        return "Seasonal Month Pattern"

    if name.startswith("day_of_year_"):
        return "Annual Seasonal Pattern"

    if name in ["day_sin", "day_cos"]:
        return "Day-of-Month Pattern"

    return name.replace("_", " ").title()


def get_model_core(model):
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        if "model" in steps:
            return steps["model"]

    return model


def get_feature_importance(model, feature_columns):
    """
    Extract feature importance from a fitted model.
    If the model has a preprocessing pipeline, use
    transformed feature names when available.
    """

    core_model = get_model_core(model)

    if not hasattr(core_model, "feature_importances_"):
        raise ValueError(
            "Feature importance is not available for this model."
        )

    importance = core_model.feature_importances_

    names = feature_columns

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")

        if preprocessor is not None:
            try:
                names = preprocessor.get_feature_names_out()
            except Exception:
                names = feature_columns

    if len(names) != len(importance):
        names = [
            f"Feature {i + 1}"
            for i in range(len(importance))
        ]

    result = pd.DataFrame({
        "Feature": names,
        "Importance": importance
    })

    result["Feature"] = result["Feature"].apply(
        friendly_feature_name
    )

    result = (
        result.groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )

    return result


# =========================================================
# 7. LOAD PROJECT INFORMATION
# =========================================================

try:
    df = load_data()
    config = load_project_config()
    models = load_models()

    DATA_LOADED = True

except Exception as error:
    DATA_LOADED = False
    st.error(f"Unable to load project files: {error}")
    st.stop()


FEATURE_COLUMNS = get_feature_columns(config)
TARGET_COLUMNS = get_target_columns(config)

city_list = sorted(df["city"].dropna().unique().tolist())

if not city_list:
    st.error("No cities were found in the dataset.")
    st.stop()


# =========================================================
# 8. MAIN HEADER
# =========================================================

st.markdown("""
<div class="brand">
    LANKAWEATHER &nbsp; / &nbsp; CLIMATE INTELLIGENCE
</div>

<div class="main-title">
    Understand the Weather.<br>
    Explore the Climate.
</div>

<div class="main-description">
    Explore weather patterns and next-day forecasting
    across multiple districts of Sri Lanka using ensemble learning.
</div>

<hr class="header-line">
""", unsafe_allow_html=True)


# =========================================================
# 9. TOP NAVIGATION
# =========================================================

PAGES = [
    "Overview",
    "Predictions",
    "Model Analysis",
    "Feature Insights",
    "Dataset Details"
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Overview"

nav_columns = st.columns(len(PAGES))

for index, page in enumerate(PAGES):

    with nav_columns[index]:

        is_selected = (
            st.session_state.selected_page == page
        )

        if st.button(
            page,
            key=f"navigation_{index}",
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_page = page
            st.rerun()


selected_page = st.session_state.selected_page


# =========================================================
# 10. OVERVIEW PAGE
# =========================================================

if selected_page == "Overview":

    section_header(
        "01 / WEATHER OVERVIEW",
        "Weather at a Glance",
        "Explore the latest available weather observations across Sri Lanka."
    )

    st.write("")

    # City selector - Ampara is the default city.

    default_city = (
        city_list.index("Ampara")
        if "Ampara" in city_list
        else 0
    )

    selected_city = st.selectbox(
        "Select a City",
        city_list,
        index=default_city,
        key="overview_city"
    )

    city_data = df[df["city"] == selected_city].copy()
    city_data = city_data.sort_values("date")

    latest = city_data.iloc[-1]

    latest_date = latest["date"].strftime("%d %B %Y")

    st.caption(
        f"Latest available observation: {latest_date}"
    )

    st.write("")

    # Latest weather metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Maximum Temperature",
            f"{latest['temp_max_c']:.1f} °C"
        )

    with col2:
        st.metric(
            "Minimum Temperature",
            f"{latest['temp_min_c']:.1f} °C"
        )

    with col3:
        st.metric(
            "Rainfall",
            f"{latest['precipitation_mm']:.1f} mm"
        )

    with col4:
        st.metric(
            "Average Humidity",
            f"{latest['humidity_mean_pct']:.1f}%"
        )

    st.write("")

    # Recent temperature trend

    st.markdown("### Recent Temperature Trend")

    recent_data = city_data.tail(90)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_data["date"],
        y=recent_data["temp_max_c"],
        name="Maximum Temperature",
        mode="lines",
        line=dict(color="#A3E635", width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=recent_data["date"],
        y=recent_data["temp_min_c"],
        name="Minimum Temperature",
        mode="lines",
        line=dict(color="#36D6C5", width=2.5)
    ))

    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,39,30,0.5)",
        font=dict(color="#E8F4EC"),
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),
        margin=dict(l=15, r=15, t=45, b=15)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rainfall trend

    st.markdown("### Recent Rainfall Pattern")

    rain_fig = px.bar(
        recent_data,
        x="date",
        y="precipitation_mm",
        labels={
            "date": "Date",
            "precipitation_mm": "Rainfall (mm)"
        }
    )

    rain_fig.update_traces(
        marker_color="#36D6C5"
    )

    rain_fig.update_layout(
        height=320,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,39,30,0.5)",
        font=dict(color="#E8F4EC"),
        margin=dict(l=15, r=15, t=25, b=15)
    )

    st.plotly_chart(
        rain_fig,
        use_container_width=True
    )


# =========================================================
# 11. WEATHER PREDICTION PAGE
# =========================================================

elif selected_page == "Predictions":

    section_header(
        "02 / WEATHER FORECAST",
        "Next-Day Weather Predictions",
        "Generate next-day weather predictions using trained ensemble learning models."
    )

    st.write("")

    if not FEATURE_COLUMNS:
        st.error(
            "Feature columns were not found in project_config.joblib. "
            "Please check the saved project configuration."
        )

    elif not models:
        st.error(
            "No prediction models could be loaded. "
            "Please check the saved model files."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            prediction_city = st.selectbox(
                "Select City",
                city_list,
                index=(
                    city_list.index("Ampara")
                    if "Ampara" in city_list
                    else 0
                ),
                key="prediction_city"
            )

        available_models = [
            name for name in PREDICTION_MODELS
            if name in models
        ]

        with col2:
            default_model_index = (
                available_models.index("XGBoost")
                if "XGBoost" in available_models
                else 0
            )

            prediction_model_name = st.selectbox(
                "Select Prediction Model",
                available_models,
                index=default_model_index,
                key="prediction_model"
            )

        try:
            input_row = get_prediction_row(
                df,
                prediction_city,
                FEATURE_COLUMNS
            )

            observation_date = input_row["date"]
            target_date = observation_date + pd.Timedelta(days=1)

            st.write("")

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.metric(
                    "Input Observation Date",
                    observation_date.strftime("%d %b %Y")
                )

            with info_col2:
                st.metric(
                    "Prediction Target Date",
                    target_date.strftime("%d %b %Y")
                )

            st.caption(
                "The prediction uses the latest available historical "
                "weather observations for the selected city."
            )

            st.write("")

            if st.button(
                "Generate Weather Prediction",
                type="primary",
                use_container_width=True
            ):

                selected_model = models[prediction_model_name]

                X_input = pd.DataFrame(
                    [input_row[FEATURE_COLUMNS].to_dict()],
                    columns=FEATURE_COLUMNS
                )

                prediction = selected_model.predict(X_input)

                prediction = np.asarray(prediction)

                if prediction.ndim == 1:
                    prediction = prediction.reshape(1, -1)

                prediction = prediction[0]

                if len(prediction) != len(TARGET_COLUMNS):
                    raise ValueError(
                        "The number of predicted values does not "
                        "match the saved target columns."
                    )

                prediction_results = dict(
                    zip(TARGET_COLUMNS, prediction)
                )

                st.markdown("### Prediction Results")

                result1, result2, result3 = st.columns(3)

                with result1:
                    st.metric(
                        "Predicted Maximum Temperature",
                        f"{prediction_results['next_day_temp_max']:.1f} °C"
                    )

                with result2:
                    st.metric(
                        "Predicted Minimum Temperature",
                        f"{prediction_results['next_day_temp_min']:.1f} °C"
                    )

                with result3:
                    st.metric(
                        "Predicted Rainfall",
                        f"{max(0, prediction_results['next_day_rainfall']):.1f} mm"
                    )

                st.success(
                    f"Prediction generated for {prediction_city} "
                    f"on {target_date.strftime('%d %B %Y')} "
                    f"using {prediction_model_name}."
                )

        except Exception as error:
            st.error(
                f"Prediction could not be generated: {error}"
            )


# =========================================================
# 12. MODEL ANALYSIS PAGE
# =========================================================

elif selected_page == "Model Analysis":

    section_header(
        "03 / MODEL PERFORMANCE",
        "Compare the Models",
        "Compare ensemble learning models using their final test-set performance."
    )

    st.write("")

    metric_choice = st.selectbox(
        "Select Evaluation Metric",
        ["MAE", "RMSE", "R²"]
    )

    if metric_choice == "MAE":
        best_row = MODEL_RESULTS.loc[
            MODEL_RESULTS["MAE"].idxmin()
        ]
        best_description = "Lowest MAE indicates lower average prediction error."

    elif metric_choice == "RMSE":
        best_row = MODEL_RESULTS.loc[
            MODEL_RESULTS["RMSE"].idxmin()
        ]
        best_description = "Lowest RMSE indicates lower large prediction errors."

    else:
        best_row = MODEL_RESULTS.loc[
            MODEL_RESULTS["R²"].idxmax()
        ]
        best_description = "Highest R² indicates the strongest explained variation."

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            f"Best Model - {metric_choice}",
            best_row["Model"],
            f"{best_row[metric_choice]:.4f}"
        )

    with col2:
        st.markdown(
            f"""
            <div class="content-panel" style="margin-top:0;">
                <div class="section-label">METRIC INTERPRETATION</div>
                <div class="section-description">
                    {best_description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Model comparison chart

    chart_data = MODEL_RESULTS.sort_values(
        metric_choice,
        ascending=(metric_choice != "R²")
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y=metric_choice,
        color="Model",
        text_auto=".4f"
    )

    fig.update_layout(
        height=430,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,39,30,0.5)",
        font=dict(color="#E8F4EC"),
        showlegend=False,
        xaxis_title="Machine Learning Model",
        yaxis_title=metric_choice,
        margin=dict(l=15, r=15, t=30, b=15)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Complete Model Performance")

    display_results = MODEL_RESULTS.copy()

    st.dataframe(
        display_results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Understanding the Evaluation Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown("**MAE**")
        st.caption(
            "Mean Absolute Error measures the average absolute difference "
            "between actual and predicted values. Lower is better."
        )

    with metric_col2:
        st.markdown("**RMSE**")
        st.caption(
            "Root Mean Squared Error gives more weight to larger prediction "
            "errors. Lower is better."
        )

    with metric_col3:
        st.markdown("**R² Score**")
        st.caption(
            "R-squared measures how much variation is explained by the model. "
            "Higher is better."
        )


# =========================================================
# 13. FEATURE INSIGHTS PAGE
# =========================================================

elif selected_page == "Feature Insights":

    section_header(
        "04 / MODEL INTERPRETATION",
        "Understand the Predictions",
        "Explore which weather and historical features influence the model."
    )

    st.write("")

    if "Random Forest" not in models:
        st.warning(
            "Random Forest model could not be loaded. "
            "Feature importance is unavailable."
        )

    else:

        try:
            importance_df = get_feature_importance(
                models["Random Forest"],
                FEATURE_COLUMNS
            )

            top_n = st.slider(
                "Number of Important Features",
                min_value=5,
                max_value=min(20, len(importance_df)),
                value=min(10, len(importance_df))
            )

            top_features = importance_df.head(top_n).sort_values(
                "Importance",
                ascending=True
            )

            fig = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                text_auto=".4f"
            )

            fig.update_traces(
                marker_color="#A3E635"
            )

            fig.update_layout(
                height=max(400, top_n * 42),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,39,30,0.5)",
                font=dict(color="#E8F4EC"),
                xaxis_title="Feature Importance",
                yaxis_title="Weather Feature",
                margin=dict(l=15, r=15, t=25, b=15)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("### Feature Importance Table")

            st.dataframe(
                importance_df.style.format({
                    "Importance": "{:.5f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Feature importance shows the relative contribution of "
                "features to the Random Forest model. It does not prove "
                "that a feature causes a weather change."
            )

        except Exception as error:
            st.error(
                f"Could not calculate feature importance: {error}"
            )


# =========================================================
# 14. DATASET DETAILS PAGE
# =========================================================

elif selected_page == "Dataset Details":

    section_header(
        "05 / DATA INFORMATION",
        "Explore the Dataset",
        "Review the dataset coverage, structure and quality information."
    )

    st.write("")

    total_records = len(df)
    total_cities = df["city"].nunique()
    start_date = df["date"].min()
    end_date = df["date"].max()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:
        st.metric(
            "Cities Covered",
            total_cities
        )

    with col3:
        st.metric(
            "Weather Variables",
            len(WEATHER_COLUMNS)
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dataset Coverage")

        st.write(f"**Start Date:** {start_date.strftime('%d %B %Y')}")
        st.write(f"**End Date:** {end_date.strftime('%d %B %Y')}")
        st.write("**Frequency:** Daily")
        st.write("**Geographical Coverage:** Sri Lanka")

    with col2:
        st.markdown("### Weather Variables")

        variable_names = [
            "Maximum Temperature (°C)",
            "Minimum Temperature (°C)",
            "Precipitation (mm)",
            "Maximum Wind Speed (km/h)",
            "Mean Relative Humidity (%)"
        ]

        for variable in variable_names:
            st.write(f"• {variable}")

    st.markdown("### Data Quality Summary")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric(
            "Missing Values",
            f"{int(df.isnull().sum().sum()):,}"
        )

    with quality_col2:
        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

    with quality_col3:
        st.metric(
            "Date Coverage",
            f"{(end_date - start_date).days + 1:,} days"
        )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset CSV",
        data=csv_data,
        file_name="sri_lanka_weather_daily.csv",
        mime="text/csv"
    )

    st.caption(
        "Dataset source: Lanka Data Tracker, based on Open-Meteo data. "
        "These are not official Sri Lankan Department of Meteorology "
        "station observations."
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("""
<hr class="header-line">

<div style="
    text-align:center;
    padding:15px 0;
    color:#AFC7B8;
    font-size:12px;
">
    <span style="color:#A3E635; font-weight:700;">
        LANKAWEATHER
    </span>
    &nbsp; | &nbsp;
    Sri Lankan Multi-District Weather Forecasting
    &nbsp; | &nbsp;
    Powered by Ensemble Learning
</div>
""", unsafe_allow_html=True)
