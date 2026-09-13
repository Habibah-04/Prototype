import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }

    .status-normal {
        color: #16a34a;
        font-weight: 700;
    }

    .status-warning {
        color: #f59e0b;
        font-weight: 700;
    }

    .status-danger {
        color: #dc2626;
        font-weight: 700;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        margin-bottom: 15px;
    }

    .anomaly-box {
        padding: 15px;
        border-radius: 10px;
        background: #fef2f2;
        border-left: 5px solid #dc2626;
        margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ SkyGuard AI")
st.caption("Automatic Weather Station Anomaly Detection System")


# ============================================================
# DEMO DATA GENERATION
# ============================================================

@st.cache_data
def generate_demo_data():

    np.random.seed(42)

    stations = [
        "AWS-001",
        "AWS-002",
        "AWS-003",
        "AWS-004",
        "AWS-005"
    ]

    start_time = datetime.now() - timedelta(hours=23)

    records = []

    station_coordinates = {
        "AWS-001": (25.3176, 82.9739),
        "AWS-002": (25.4358, 81.8463),
        "AWS-003": (26.4499, 80.3319),
        "AWS-004": (25.5941, 85.1376),
        "AWS-005": (26.8467, 80.9462)
    }

    for station in stations:

        lat, lon = station_coordinates[station]

        for hour in range(24):

            timestamp = start_time + timedelta(hours=hour)

            temperature = np.random.normal(30, 3)
            humidity = np.random.normal(60, 8)
            pressure = np.random.normal(1012, 5)

            records.append({
                "Station": station,
                "Timestamp": timestamp,
                "Temperature": round(temperature, 2),
                "Humidity": round(humidity, 2),
                "Pressure": round(pressure, 2),
                "Latitude": lat,
                "Longitude": lon
            })

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Intentional anomaly for demonstration
    # --------------------------------------------------------

    anomaly_index = (
        (df["Station"] == "AWS-002") &
        (df["Timestamp"].dt.hour == df["Timestamp"].dt.hour.max())
    )

    if anomaly_index.any():
        index = df[anomaly_index].index[0]
        df.loc[index, "Temperature"] = 47.0

    return df


df = generate_demo_data()


# ============================================================
# MACHINE LEARNING ANOMALY DETECTION
# ============================================================

def run_ml_detection(data):

    result = data.copy()

    # Only Temperature, Humidity and Pressure are used
    features = [
        "Temperature",
        "Humidity",
        "Pressure"
    ]

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=150
    )

    result["ML Prediction"] = model.fit_predict(
        result[features]
    )

    result["ML Anomaly"] = result["ML Prediction"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    result["Anomaly Score"] = model.decision_function(
        result[features]
    )

    return result


df_ml = run_ml_detection(df)


# ============================================================
# RULE-BASED DETECTION
# ============================================================

def run_rule_detection(data):

    result = data.copy()

    statuses = []
    reasons_list = []

    for _, current in result.iterrows():

        reasons = []

        # Temperature validation
        if current["Temperature"] < -20:
            reasons.append("Very low temperature")

        if current["Temperature"] > 45:
            reasons.append("Very high temperature")

        # Humidity validation
        if current["Humidity"] < 0:
            reasons.append("Invalid humidity")

        if current["Humidity"] > 100:
            reasons.append("Invalid humidity")

        # Pressure validation
        if current["Pressure"] < 850:
            reasons.append("Very low pressure")

        if current["Pressure"] > 1100:
            reasons.append("Very high pressure")

        if reasons:
            statuses.append("Anomaly")
            reasons_list.append(", ".join(reasons))
        else:
            statuses.append("Normal")
            reasons_list.append("")

    result["Rule Status"] = statuses
    result["Rule Reason"] = reasons_list

    return result


df_final = run_rule_detection(df_ml)


# ============================================================
# COMBINED ANOMALY STATUS
# ============================================================

df_final["Final Status"] = np.where(
    (df_final["ML Anomaly"] == "Anomaly") |
    (df_final["Rule Status"] == "Anomaly"),
    "Anomaly",
    "Normal"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛰️ SkyGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Anomaly Detection",
        "Sensor Analytics",
        "Station Map",
        "Data Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **AI-Powered AWS Monitoring**

    Parameters monitored:

    🌡️ Temperature  
    💧 Humidity  
    🌬️ Pressure
    """
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">📊 System Dashboard</div>',
        unsafe_allow_html=True
    )

    total_stations = df_final["Station"].nunique()

    total_readings = len(df_final)

    total_anomalies = (
        df_final["Final Status"] == "Anomaly"
    ).sum()

    normal_readings = total_readings - total_anomalies

    anomaly_rate = (
        total_anomalies / total_readings * 100
        if total_readings > 0
        else 0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "AWS Stations",
            total_stations
        )

    with col2:
        st.metric(
            "Total Readings",
            total_readings
        )

    with col3:
        st.metric(
            "Anomalies",
            total_anomalies
        )

    with col4:
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Latest Readings
    # --------------------------------------------------------

    st.subheader("📡 Latest Station Readings")

    latest = (
        df_final
        .sort_values("Timestamp")
        .groupby("Station")
        .tail(1)
        .sort_values("Station")
    )

    display_df = latest[
        [
            "Station",
            "Timestamp",
            "Temperature",
            "Humidity",
            "Pressure",
            "Final Status"
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Parameter Overview
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Temperature",
            f"{df_final['Temperature'].mean():.1f} °C"
        )

    with col2:
        st.metric(
            "Avg Humidity",
            f"{df_final['Humidity'].mean():.1f} %"
        )

    with col3:
        st.metric(
            "Avg Pressure",
            f"{df_final['Pressure'].mean():.1f} hPa"
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.markdown(
        '<div class="section-title">🚨 Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        SkyGuard AI combines <b>Isolation Forest machine learning</b>
        with rule-based validation to identify abnormal AWS readings.
        </div>
        """,
        unsafe_allow_html=True
    )

    anomalies = df_final[
        df_final["Final Status"] == "Anomaly"
    ].copy()

    st.subheader(
        f"Detected Anomalies: {len(anomalies)}"
    )

    if len(anomalies) == 0:

        st.success("✅ No anomalies detected.")

    else:

        for _, row in anomalies.iterrows():

            reason = row["Rule Reason"]

            if not reason:
                reason = "Detected by AI anomaly model"

            st.markdown(
                f"""
                <div class="anomaly-box">
                    <b>🚨 {row['Station']}</b><br>
                    Time: {row['Timestamp']}<br>
                    Temperature: {row['Temperature']} °C<br>
                    Humidity: {row['Humidity']} %<br>
                    Pressure: {row['Pressure']} hPa<br>
                    Reason: {reason}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader("🔍 Detection Method")

    method_col1, method_col2 = st.columns(2)

    with method_col1:

        st.markdown(
            """
            ### 🤖 Machine Learning

            **Isolation Forest**

            Uses the following parameters:

            - Temperature
            - Humidity
            - Pressure

            The model identifies observations that differ
            significantly from normal sensor patterns.
            """
        )

    with method_col2:

        st.markdown(
            """
            ### 📏 Rule-Based Validation

            Checks sensor values against predefined limits:

            - Temperature: -20°C to 45°C
            - Humidity: 0% to 100%
            - Pressure: 850 to 1100 hPa

            Values outside these limits are flagged as anomalies.
            """
        )


# ============================================================
# SENSOR ANALYTICS
# ============================================================

elif page == "Sensor Analytics":

    st.markdown(
        '<div class="section-title">📈 Sensor Analytics</div>',
        unsafe_allow_html=True
    )

    parameter = st.selectbox(
        "Select Parameter",
        [
            "Temperature",
            "Humidity",
            "Pressure"
        ]
    )

    station = st.selectbox(
        "Select Station",
        sorted(df_final["Station"].unique())
    )

    station_data = df_final[
        df_final["Station"] == station
    ].sort_values("Timestamp")

    # --------------------------------------------------------
    # Line Chart
    # --------------------------------------------------------

    fig = px.line(
        station_data,
        x="Timestamp",
        y=parameter,
        markers=True,
        title=f"{parameter} Trend — {station}"
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=parameter,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    st.subheader("📊 Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Minimum",
            f"{station_data[parameter].min():.2f}"
        )

    with c2:
        st.metric(
            "Maximum",
            f"{station_data[parameter].max():.2f}"
        )

    with c3:
        st.metric(
            "Average",
            f"{station_data[parameter].mean():.2f}"
        )

    with c4:
        st.metric(
            "Std. Deviation",
            f"{station_data[parameter].std():.2f}"
        )


# ============================================================
# STATION MAP
# ============================================================

elif page == "Station Map":

    st.markdown(
        '<div class="section-title">🗺️ AWS Station Map</div>',
        unsafe_allow_html=True
    )

    latest = (
        df_final
        .sort_values("Timestamp")
        .groupby("Station")
        .tail(1)
    )

    map_data = latest[
        [
            "Station",
            "Latitude",
            "Longitude",
            "Temperature",
            "Humidity",
            "Pressure",
            "Final Status"
        ]
    ].copy()

    st.map(
        map_data,
        latitude="Latitude",
        longitude="Longitude"
    )

    st.subheader("📍 Station Status")

    st.dataframe(
        map_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="section-title">🗃️ Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore the complete AWS sensor dataset."
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_station = st.multiselect(
            "Filter by Station",
            sorted(df_final["Station"].unique()),
            default=sorted(df_final["Station"].unique())
        )

    with col2:

        selected_status = st.multiselect(
            "Filter by Status",
            ["Normal", "Anomaly"],
            default=["Normal", "Anomaly"]
        )

    filtered_df = df_final[
        df_final["Station"].isin(selected_station) &
        df_final["Final Status"].isin(selected_status)
    ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="aws_shield_sensor_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌦️ SkyGuard AI | AI-powered weather station anomaly detection"
)
