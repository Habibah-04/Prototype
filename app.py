import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .danger-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
    }

    .warning-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #fef3c7;
        border-left: 6px solid #d97706;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# GENERATE DEMO AWS DATA
# =========================================================

@st.cache_data
def generate_demo_data():

    stations = {
        "AWS-001": {
            "latitude": 27.1767,
            "longitude": 78.0081,
            "temperature": 29
        },
        "AWS-002": {
            "latitude": 26.8467,
            "longitude": 80.9462,
            "temperature": 29
        },
        "AWS-003": {
            "latitude": 25.4358,
            "longitude": 81.8463,
            "temperature": 30
        },
        "AWS-004": {
            "latitude": 26.4499,
            "longitude": 80.3319,
            "temperature": 28
        },
        "AWS-005": {
            "latitude": 25.3176,
            "longitude": 82.9739,
            "temperature": 29
        }
    }

    rng = np.random.default_rng(42)

    data = []

    for station, info in stations.items():

        base_temperature = info["temperature"]

        for hour in range(24):

            temperature = (
                base_temperature
                + 1.2 * np.sin(hour / 24 * 2 * np.pi)
                + rng.normal(0, 0.35)
            )

            humidity = (
                70
                - (temperature - base_temperature) * 2
                + rng.normal(0, 1.5)
            )

            pressure = 1008 + rng.normal(0, 2)

            # -------------------------------------------------
            # INTENTIONAL ANOMALY FOR SIH DEMO
            # -------------------------------------------------

            if station == "AWS-002" and hour == 23:
                temperature = 47.0

            data.append({
                "Station": station,
                "Hour": hour,
                "Temperature": round(temperature, 2),
                "Humidity": round(humidity, 2),
                "Pressure": round(pressure, 2),
                "Latitude": info["latitude"],
                "Longitude": info["longitude"]
            })

    return pd.DataFrame(data)


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

def run_ml_detection(df):

    features = [
        "Temperature",
        "Humidity",
        "Pressure"
    ]

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    )

    model.fit(df[features])

    result = df.copy()

    result["ML Prediction"] = model.predict(
        result[features]
    )

    result["ML Score"] = -model.decision_function(
        result[features]
    )

    return result, model


# =========================================================
# RULE BASED DETECTION
# =========================================================

def rule_based_detection(current, previous):

    reasons = []

    if len(previous) >= 3:

        previous_average = (
            previous["Temperature"]
            .tail(3)
            .mean()
        )

        temperature_difference = abs(
            current["Temperature"]
            - previous_average
        )

        if temperature_difference >= 8:

            reasons.append(
                "Sudden temperature spike"
            )

    # Physical range validation

    if (
        current["Temperature"] < -10
        or current["Temperature"] > 50
    ):

        reasons.append(
            "Temperature outside expected range"
        )

    if (
        current["Humidity"] < 0
        or current["Humidity"] > 100
    ):

        reasons.append(
            "Humidity outside valid range"
        )

    return reasons


# =========================================================
# NEARBY STATION CHECK
# =========================================================

def nearby_station_check(current, df):

    same_hour = df[
        df["Hour"] == current["Hour"]
    ]

    other_stations = same_hour[
        same_hour["Station"]
        != current["Station"]
    ]

    if other_stations.empty:

        return True, "No nearby comparison data available."

    median_temperature = (
        other_stations["Temperature"]
        .median()
    )

    difference = abs(
        current["Temperature"]
        - median_temperature
    )

    if difference >= 8:

        return (
            False,
            f"Nearby stations are around "
            f"{median_temperature:.1f}°C."
        )

    return (
        True,
        f"Nearby stations are around "
        f"{median_temperature:.1f}°C."
    )


# =========================================================
# COMPLETE ANOMALY ANALYSIS
# =========================================================

def analyze_station(df, station):

    station_data = (
        df[df["Station"] == station]
        .sort_values("Hour")
        .reset_index(drop=True)
    )

    current = station_data.iloc[-1]

    previous = station_data.iloc[:-1]

    # Rule detection

    rule_reasons = rule_based_detection(
        current,
        previous
    )

    # Nearby station detection

    nearby_ok, nearby_message = (
        nearby_station_check(
            current,
            df
        )
    )

    # ML detection

    ml_anomaly = (
        current["ML Prediction"] == -1
    )

    reasons = []

    reasons.extend(rule_reasons)

    if not nearby_ok:

        reasons.append(
            "Reading differs significantly "
            "from nearby stations"
        )

    if ml_anomaly:

        reasons.append(
            "ML model detected an unusual pattern"
        )

    anomaly_detected = len(reasons) > 0

    # Classification

    if anomaly_detected:

        if (
            not nearby_ok
            and len(rule_reasons) > 0
        ):

            classification = (
                "Likely Faulty Sensor Reading"
            )

            recommendation = (
                "Inspect and recalibrate "
                "the temperature sensor."
            )

        else:

            classification = (
                "Suspicious Observation"
            )

            recommendation = (
                "Verify this reading using "
                "historical and nearby station data."
            )

    else:

        classification = (
            "Normal Observation"
        )

        recommendation = (
            "No immediate action required."
        )

    # Explainable confidence

    if len(reasons) >= 3:

        confidence = 98

    elif len(reasons) == 2:

        confidence = 92

    elif len(reasons) == 1:

        confidence = 85

    else:

        confidence = 12

    return {
        "current": current,
        "previous": previous,
        "rule_reasons": rule_reasons,
        "nearby_ok": nearby_ok,
        "nearby_message": nearby_message,
        "ml_anomaly": ml_anomaly,
        "reasons": reasons,
        "anomaly": anomaly_detected,
        "classification": classification,
        "recommendation": recommendation,
        "confidence": confidence
    }


# =========================================================
# LOAD DATA
# =========================================================

df = generate_demo_data()

df, ml_model = run_ml_detection(df)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ SkyGuard AI")

st.sidebar.caption(
    "AI/ML-Based Intelligent Anomaly Detection"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Anomaly Detection",
        "📈 Sensor Analytics",
        "🗺️ Station Map",
        "📋 Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    Prototype Mode

    The application currently uses
    simulated AWS observations for
    demonstration.

    Real AWS/API/database data can
    be connected later.
    """
)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🛡️ SkyGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect the Data Error Before It Becomes a Forecast Error.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    latest = (
        df.sort_values("Hour")
        .groupby("Station")
        .tail(1)
        .copy()
    )

    anomaly_stations = []

    for station in latest["Station"]:

        analysis = analyze_station(
            df,
            station
        )

        if analysis["anomaly"]:

            anomaly_stations.append(
                station
            )

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Stations",
        len(latest)
    )

    col2.metric(
        "Normal",
        len(latest)
        - len(anomaly_stations)
    )

    col3.metric(
        "Anomalies",
        len(anomaly_stations)
    )

    col4.metric(
        "Critical",
        sum(
            station == "AWS-002"
            for station in anomaly_stations
        )
    )

    st.divider()

    # Station status

    st.subheader(
        "📡 Live Station Status"
    )

    display = latest[
        [
            "Station",
            "Temperature",
            "Humidity",
                "Pressure"
        ]
    ].copy()

    display["Status"] = [
        (
            "🔴 Anomaly"
            if station in anomaly_stations
            else "🟢 Normal"
        )
        for station in display["Station"]
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    # Temperature chart

    st.subheader(
        "🌡️ Current Temperature"
    )

    fig = px.bar(
        latest,
        x="Station",
        y="Temperature",
        color="Station",
        text="Temperature",
        title="Latest Temperature by AWS Station"
    )

    fig.update_traces(
        texttemplate="%{text}°C",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Alert

    if anomaly_stations:

        st.error(
            "⚠️ Anomaly detected at: "
            + ", ".join(anomaly_stations)
            + ". Open Anomaly Detection "
            "for detailed analysis."
        )

# =========================================================
# ANOMALY DETECTION
# =========================================================

elif page == "🔍 Anomaly Detection":

    st.subheader(
        "🔍 Intelligent Anomaly Detection"
    )

    station = st.selectbox(
        "Select AWS Station",
        sorted(
            df["Station"].unique()
        )
    )

    analysis = analyze_station(
        df,
        station
    )

    current = analysis["current"]

    previous = analysis["previous"]

    # Top metrics

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Temperature",
        f"{current['Temperature']:.1f} °C"
    )

    col2.metric(
        "Anomaly Confidence",
        f"{analysis['confidence']}%"
    )

    col3.metric(
        "Classification",
        analysis["classification"]
    )

    st.divider()

    # Historical graph

    st.subheader(
        "📈 Sensor Reading History"
    )

    chart_data = previous[
        ["Hour", "Temperature"]
    ].copy()

    latest_point = pd.DataFrame({
        "Hour": [current["Hour"]],
        "Temperature": [
            current["Temperature"]
        ]
    })

    chart_data = pd.concat(
        [
            chart_data,
            latest_point
        ],
        ignore_index=True
    )

    fig = px.line(
        chart_data,
        x="Hour",
        y="Temperature",
        markers=True,
        title=f"{station} Temperature History"
    )

    fig.add_hline(
        y=40,
        line_dash="dash",
        annotation_text="High Temperature Alert"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Explainable AI

    st.subheader(
        "🧠 Explainable AI Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 1. Rule-Based Validation"
        )

        if analysis["rule_reasons"]:

            st.error(
                "Anomaly detected"
            )

            for reason in analysis[
                "rule_reasons"
            ]:

                st.write(
                    "• " + reason
                )

        else:

            st.success(
                "No rule-based anomaly"
            )

        st.markdown(
            "### 2. Historical Comparison"
        )

        if len(previous) >= 3:

            average = (
                previous[
                    "Temperature"
                ]
                .tail(3)
                .mean()
            )

            difference = abs(
                current["Temperature"]
                - average
            )

            st.write(
                f"Previous 3-reading average: "
                f"**{average:.1f}°C**"
            )

            st.write(
                f"Current reading: "
                f"**{current['Temperature']:.1f}°C**"
            )

            st.write(
                f"Difference: "
                f"**{difference:.1f}°C**"
            )

    with col2:

        st.markdown(
            "### 3. Nearby Station Validation"
        )

        if analysis["nearby_ok"]:

            st.success(
                "Nearby stations are consistent"
            )

        else:

            st.error(
                "Nearby station mismatch"
            )

        st.write(
            analysis["nearby_message"]
        )

        st.markdown(
            "### 4. ML Detection"
        )

        if analysis["ml_anomaly"]:

            st.error(
                "ML model flagged this observation"
            )

        else:

            st.success(
                "ML model did not flag this observation"
            )

    st.divider()

    # Final classification

    if (
        analysis["classification"]
        == "Likely Faulty Sensor Reading"
    ):

        st.markdown(
            f"""
            <div class="danger-box">

            <h2>🔴 Likely Faulty Sensor Reading</h2>

            <p>
            The system detected multiple indicators
            suggesting that this observation may be
            caused by a sensor/data problem.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    elif (
        analysis["classification"]
        == "Suspicious Observation"
    ):

        st.markdown(
            f"""
            <div class="warning-box">

            <h2>🟠 Suspicious Observation</h2>

            <p>
            The observation requires verification.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="success-box">

            <h2>🟢 Normal Observation</h2>

            <p>
            No significant anomaly indicators were detected.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader(
        "Why is this reading suspicious?"
    )

    if analysis["reasons"]:

        for reason in analysis["reasons"]:

            st.write(
                "🔎 " + reason
            )

    else:

        st.write(
            "No anomaly indicators detected."
        )

    st.warning(
        "Recommended Action: "
        + analysis["recommendation"]
    )

# =========================================================
# SENSOR ANALYTICS
# =========================================================

elif page == "📈 Sensor Analytics":

    st.subheader(
        "📈 Sensor Analytics"
    )

    station = st.selectbox(
        "Select Station",
        sorted(
            df["Station"].unique()
        ),
        key="analytics_station"
    )

    parameter = st.selectbox(
        "Select Parameter",
        [
            "Temperature",
            "Humidity",
                "Pressure"
        ]
    )

    station_data = (
        df[df["Station"] == station]
        .sort_values("Hour")
    )

    fig = px.line(
        station_data,
        x="Hour",
        y=parameter,
        markers=True,
        title=f"{station} - {parameter}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "📊 Statistical Summary"
    )

    summary = (
        station_data[
            [parameter]
        ]
        .describe()
        .round(2)
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

# =========================================================
# STATION MAP
# =========================================================

elif page == "🗺️ Station Map":

    st.subheader(
        "🗺️ Automatic Weather Station Network"
    )

    latest = (
        df.sort_values("Hour")
        .groupby("Station")
        .tail(1)
        .copy()
    )

    latest["Status"] = [
        (
            "Anomaly"
            if analyze_station(
                df,
                station
            )["anomaly"]
            else "Normal"
        )
        for station in latest["Station"]
    ]

    map_data = latest[
        [
            "Latitude",
            "Longitude"
        ]
    ].rename(
        columns={
            "Latitude": "lat",
            "Longitude": "lon"
        }
    )

    st.map(
        map_data,
        zoom=6
    )

    st.subheader(
        "Station Information"
    )

    st.dataframe(
        latest[
            [
                "Station",
                "Latitude",
                "Longitude",
                "Temperature",
                "Humidity",
                "Status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "📋 Data Explorer":

    st.subheader(
        "📋 AWS Observation Data"
    )

    uploaded_file = st.file_uploader(
        "Upload AWS CSV Data",
        type=["csv"]
    )

    if uploaded_file:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            st.success(
                "CSV uploaded successfully."
            )

            st.dataframe(
                uploaded_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )

    else:

        st.info(
            "Showing simulated AWS data."
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "⬇️ Download Demo AWS Data",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="skyguard_ai_demo_data.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SkyGuard AI Prototype | SIH26073 | "
    "AI/ML-Based Intelligent Anomaly Detection "
    "for Automatic Weather Stations"
)

st.caption(
    "Prototype demonstration uses simulated AWS observations."
)