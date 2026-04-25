import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

BASE = ""

st.set_page_config(page_title="Smart Logistics Analytics", page_icon="🚛", layout="wide")


def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**👥 Team — Group 82**")
    col2.markdown("Mohammad Kaif | Harshita Hoiyani | Ansh Mittal")
    col3.markdown("UCF 439 | Capstone Project | JAN-MAY 2026")


@st.cache_resource
def load_model():
    model      = joblib.load(BASE + "Models/xgboost_model.pkl")
    le_route   = joblib.load(BASE + "Models/le_route.pkl")
    le_truck   = joblib.load(BASE + "Models/le_truck_type.pkl")
    le_weather = joblib.load(BASE + "Models/le_weather_condition.pkl")
    le_road    = joblib.load(BASE + "Models/le_road_condition.pkl")
    with open(BASE + "Models/model_config.json") as f:
        config = json.load(f)
    return model, le_route, le_truck, le_weather, le_road, config


@st.cache_data
def load_data():
    df            = pd.read_csv(BASE + "data/india_logistics_v2_processed.csv")
    best_routes   = pd.read_csv(BASE + "data/best_routes.csv")
    route_metrics = pd.read_csv(BASE + "data/route_metrics_final.csv")
    return df, best_routes, route_metrics


model, le_route, le_truck, le_weather, le_road, config = load_model()
df, best_routes, route_metrics = load_data()

# Sidebar
st.sidebar.title("🚛 Smart Logistics")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Team — Group 82")
st.sidebar.markdown("""
🔹 Mohammad Kaif  
🔹 Harshita Hoiyani  
🔹 Ansh Mittal  
""")
st.sidebar.markdown("---")
st.sidebar.caption("UCF 439 | Capstone | JAN-MAY 2026")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔮 Delay Predictor",
    "🗺️ Route Recommender",
    "💡 Insights"
])

# PAGE 1: DASHBOARD
if page == "📊 Dashboard":
    st.title("🚛 Smart Logistics Analytics System")
    st.markdown("**India Heavy Freight — Performance Dashboard**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments",  f"{len(df):,}")
    col2.metric("Delayed",          f"{df['Delay_Flag'].sum():,}",
                                    f"{df['Delay_Flag'].mean()*100:.1f}%")
    col3.metric("Avg Freight Cost", f"Rs.{df['Freight_Cost_INR'].mean():,.0f}")
    col4.metric("Avg Distance",     f"{df['Distance_km'].mean():,.0f} km")
    st.markdown("---")

    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delay Rate by Truck Type (%)")
        truck_delay = df.groupby("Truck_Type")["Delay_Flag"].mean() * 100
        truck_delay = truck_delay.round(1)
        st.bar_chart(truck_delay)
    with col2:
        st.subheader("Monthly Delay Trend (%)")
        monthly = df.groupby("Month_Name")["Delay_Flag"].mean() * 100
        monthly = monthly.reindex(month_order).dropna()
        st.line_chart(monthly)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Most Delayed Routes (%)")
        route_delay = df.groupby("Route")["Delay_Flag"].mean() * 100
        st.bar_chart(route_delay.sort_values(ascending=False).head(5))
    with col2:
        st.subheader("Shipments by Truck Type")
        st.bar_chart(df["Truck_Type"].value_counts())

    show_footer()

# PAGE 2: DELAY PREDICTOR
elif page == "🔮 Delay Predictor":
    st.title("🔮 Delivery Delay Predictor")
    st.markdown("Enter shipment details to predict delay risk")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Shipment Details")
        route      = st.selectbox("Route", sorted(df["Route"].unique()))
        truck_type = st.selectbox("Truck Type", ["Heavy Truck","Medium Truck","Light Van"])
        distance   = st.number_input("Distance (km)", 100, 2000, value=500, step=50)
        weight     = st.number_input("Weight (kg)", 500, 40000, value=5000, step=500)

    with col2:
        st.subheader("Cost and Load")
        freight_cost  = st.number_input("Freight Cost (Rs.)", 5000, 500000, value=50000, step=1000)
        cost_per_km   = round(freight_cost / distance, 2)
        st.metric("Cost per KM", f"Rs.{cost_per_km}")
        load_util     = st.slider("Load Utilization (%)", 55, 100, 75)
        expected_days = st.number_input("Expected Delivery Days", 1, 5, value=2)

    with col3:
        st.subheader("Conditions")
        month      = st.selectbox("Month", list(range(1, 13)),
                        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                               "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        weather    = st.selectbox("Weather", ["Clear","Rain","Fog","Storm"])
        road       = st.selectbox("Road Condition", ["Good","Average","Poor"])
        driver_exp = st.slider("Driver Experience (years)", 1, 20, 5)

    st.markdown("---")

    if st.button("🔮 PREDICT DELIVERY STATUS", use_container_width=True):

        season_map = {
            12:"Winter", 1:"Winter",  2:"Winter",
            3:"Summer",  4:"Summer",  5:"Summer",
            6:"Monsoon", 7:"Monsoon", 8:"Monsoon", 9:"Monsoon",
            10:"Autumn", 11:"Autumn"
        }
        season      = season_map[month]
        is_festival = 1 if month in [1, 3, 4, 8, 10, 11] else 0
        overload    = 1 if load_util > 90 else 0
        old_truck   = 0
        high_cong   = 0
        long_route  = 1 if distance > 800 else 0
        heavy_load  = 1 if weight > 15000 else 0
        driver_age  = 30
        truck_age   = 3
        congestion  = 40
        num_stops   = 2

        try:
            le_cargo   = joblib.load(BASE + "Models/le_cargo_type.pkl")
            le_exp     = joblib.load(BASE + "Models/le_exp_category.pkl")
            le_season  = joblib.load(BASE + "Models/le_season.pkl")
            cargo_enc  = le_cargo.transform(["FMCG"])[0]
            exp_cat    = "Junior" if driver_exp < 3 else "Mid" if driver_exp < 8 else "Senior"
            exp_enc    = le_exp.transform([exp_cat])[0]
            season_enc = le_season.transform([season])[0]
        except Exception:
            cargo_enc  = 0
            exp_enc    = 1
            season_enc = 0

        input_data = np.array([[
            distance, weight, freight_cost, cost_per_km,
            expected_days, load_util, driver_exp,
            driver_age, truck_age, congestion, num_stops,
            is_festival, month,
            le_route.transform([route])[0],
            le_truck.transform([truck_type])[0],
            le_weather.transform([weather])[0],
            le_road.transform([road])[0],
            cargo_enc, season_enc,
            overload, old_truck, high_cong,
            long_route, heavy_load
        ]])

        proba      = model.predict_proba(input_data)[0][1]
        prediction = int(proba >= config["threshold"])

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("⚠️ LIKELY DELAYED")
                st.markdown(f"### Delay Probability: {proba*100:.1f}%")
            else:
                st.success("✅ ON TIME DELIVERY")
                st.markdown(f"### On-Time Probability: {(1-proba)*100:.1f}%")

        with col2:
            st.info(f"**Route:** {route}")
            st.info(f"**Truck:** {truck_type}")
            st.info(f"**Distance:** {distance} km")

        with col3:
            st.info(f"**Weight:** {weight:,} kg")
            st.info(f"**Cost/km:** Rs.{cost_per_km}")
            st.info(f"**Weather:** {weather}")

    show_footer()

# PAGE 3: ROUTE RECOMMENDER
elif page == "🗺️ Route Recommender":
    st.title("🗺️ Best Route Recommender")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        selected_month = st.selectbox("Select Month",
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
        st.markdown("---")
        st.subheader(f"Best Route for {selected_month}")
        month_data = best_routes[best_routes["Month"] == selected_month]
        if not month_data.empty:
            row = month_data.iloc[0]
            st.success(f"### 🏆 {row['Best_Route']}")
            st.metric("Delay Rate",  f"{row['Delay_%']:.1f}%")
            st.metric("Cost per KM", f"Rs.{row['Cost_per_km']:.0f}")
            st.metric("Avg Days",    f"{row['Avg_Days']:.1f}")
            st.metric("Score",       f"{row['Score']:.4f}")

    with col2:
        st.subheader("All Routes Ranked")
        if "Route" in route_metrics.columns:
            display_df = route_metrics.set_index("Route")[
                ["Rank","Delay_Rate_%","Avg_Cost_per_km",
                 "Avg_Delivery_Days","Efficiency_Score"]]
        else:
            display_df = route_metrics[
                ["Rank","Delay_Rate_%","Avg_Cost_per_km",
                 "Avg_Delivery_Days","Efficiency_Score"]]
        st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Best Route Table")
    st.dataframe(best_routes, use_container_width=True)

    show_footer()

# PAGE 4: INSIGHTS
elif page == "💡 Insights":
    st.title("💡 Key Business Insights")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 What Causes Delays?")
        st.markdown("""
        1. **Traffic Congestion** — Biggest delay driver
        2. **Storm Weather** — 15%+ higher delays
        3. **Overloaded Trucks** — Load >90% is risky
        4. **Old Trucks (>8 yrs)** — Maintenance issues
        5. **Festival Months** — Oct/Nov/Jan demand spikes
        6. **Poor Road Condition** — Highway quality matters
        7. **Long Distance (>800km)** — More risk exposure
        8. **Inexperienced Drivers** — <3 yrs = higher risk
        """)

        st.subheader("🏆 Best Routes")
        st.markdown("""
        - **#1 Ahmedabad-Mumbai** — 17% delay | Rs.84/km
        - **#2 Kolkata-Bhubaneswar** — 19% delay | Rs.87/km
        - **#3 Delhi-Amritsar** — 19% delay | Rs.84/km
        """)

    with col2:
        st.subheader("❌ Worst Routes")
        st.markdown("""
        - **#15 Delhi-Mumbai** — 30% delay | Rs.88/km
        - **#14 Mumbai-Chennai** — 30% delay | Rs.85/km
        - **#13 Chennai-Kolkata** — 32% delay | Rs.86/km
        """)

        st.subheader("📅 Seasonal Patterns")
        st.markdown("""
        - **June** — Worst month (highest delays) — Monsoon
        - **February** — Best month (lowest delays)
        - **October** — Best for Bangalore-Hyderabad (0% delay)
        """)

    st.markdown("---")
    st.subheader("🤖 ML Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",            "XGBoost")
    col2.metric("Recall",           "85.6%")
    col3.metric("Training Samples", "37,518")
    col4.metric("Features Used",    "24")

    show_footer()
