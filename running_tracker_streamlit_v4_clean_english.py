
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, datetime, timedelta
import io
import math
import calendar

DATA_FILE = Path("running_tracker_data.csv")

st.set_page_config(
    page_title="Running & Weight Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Data utilities
# ============================================================

REQUIRED_COLUMNS = [
    "date", "ran", "distance_km", "duration_min", "pace_min_per_km",
    "weight_kg", "calories_kcal", "source", "notes"
]

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS].copy()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        df["ran"] = df["ran"].fillna(False).astype(bool)

        for col in ["distance_km", "duration_min", "pace_min_per_km", "weight_kg", "calories_kcal"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["source"] = df["source"].fillna("")
        df["notes"] = df["notes"].fillna("")

    return df

def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        return normalize_df(pd.read_csv(DATA_FILE))
    return pd.DataFrame(columns=REQUIRED_COLUMNS)

def save_data(df: pd.DataFrame):
    df = normalize_df(df)
    df = df.sort_values("date")
    df.to_csv(DATA_FILE, index=False)

def calculate_pace(distance_km, duration_min):
    if pd.notna(distance_km) and pd.notna(duration_min) and distance_km > 0 and duration_min > 0:
        return duration_min / distance_km
    return np.nan

def pace_to_text(pace):
    if pd.isna(pace) or pace <= 0 or math.isinf(pace):
        return "-"
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"

def duration_to_text(duration_min):
    if pd.isna(duration_min) or duration_min <= 0:
        return "-"
    total_seconds = int(round(duration_min * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def parse_duration_input(text: str) -> float:
    text = str(text).strip()
    if ":" in text:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 2:
            return parts[0] + parts[1] / 60
        if len(parts) == 3:
            return parts[0] * 60 + parts[1] + parts[2] / 60
        raise ValueError("Use MM:SS, HH:MM:SS, or total minutes.")
    return float(text)

def estimate_calories(distance_km, weight_kg, ran=True):
    """
    Practical estimate for running:
    calories ≈ body weight (kg) × distance (km)

    This is only an estimate. If Apple Health provides calories, the app uses Apple Health data.
    """
    if pd.isna(distance_km) or pd.isna(weight_kg) or distance_km <= 0 or weight_kg <= 0:
        return np.nan
    factor = 1.0 if ran else 0.7
    return round(distance_km * weight_kg * factor, 1)

def latest_weight_or_start(df: pd.DataFrame, start_weight: float) -> float:
    df = normalize_df(df)
    weights = df["weight_kg"].dropna()
    if len(weights):
        return float(weights.iloc[-1])
    return float(start_weight)

def upsert_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_df(existing)
    new_rows = normalize_df(new_rows)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    priority = {"Apple Health": 4, "Manual": 3, "CSV Restore": 2, "": 1}
    combined["_priority"] = combined["source"].map(priority).fillna(1)
    combined = combined.sort_values(["date", "_priority"])

    out = []
    for d, group in combined.groupby("date"):
        row = {"date": d}
        row["ran"] = bool(group["ran"].fillna(False).iloc[-1])
        for col in ["distance_km", "duration_min", "pace_min_per_km", "weight_kg", "calories_kcal", "source", "notes"]:
            values = group[col].replace("", np.nan).dropna()
            row[col] = values.iloc[-1] if len(values) else np.nan
        out.append(row)

    return normalize_df(pd.DataFrame(out)).sort_values("date").reset_index(drop=True)

def fill_missing_calories(df, start_weight):
    df = normalize_df(df).sort_values("date").reset_index(drop=True)
    for idx, row in df.iterrows():
        if row["ran"] and (pd.isna(row["calories_kcal"]) or row["calories_kcal"] <= 0):
            weight = row["weight_kg"]
            if pd.isna(weight) or weight <= 0:
                prev = df.loc[:idx, "weight_kg"].dropna()
                weight = prev.iloc[-1] if len(prev) else start_weight
            df.at[idx, "calories_kcal"] = estimate_calories(row["distance_km"], weight, True)
    return df

def current_streak(df):
    df = normalize_df(df)
    ran_dates = set(df.loc[df["ran"] == True, "date"])
    d = date.today()
    streak = 0
    while d in ran_dates:
        streak += 1
        d -= timedelta(days=1)
    return streak

def longest_streak(df):
    df = normalize_df(df)
    ran_dates = sorted(set(df.loc[df["ran"] == True, "date"]))
    if not ran_dates:
        return 0
    best = cur = 1
    for prev, curr in zip(ran_dates, ran_dates[1:]):
        if curr == prev + timedelta(days=1):
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return max(best, cur)

def monthly_filter(df):
    dfp = normalize_df(df)
    if dfp.empty:
        return dfp
    today = date.today()
    return dfp[(pd.to_datetime(dfp["date"]).dt.year == today.year) & (pd.to_datetime(dfp["date"]).dt.month == today.month)]

def weekly_filter(df):
    dfp = normalize_df(df)
    if dfp.empty:
        return dfp
    today = pd.Timestamp.today().date()
    week_start = today - timedelta(days=today.weekday())
    return dfp[dfp["date"] >= week_start]

def get_weight_progress(df, start_weight, target_weight):
    df = normalize_df(df)
    weights = df["weight_kg"].dropna()
    if len(weights) == 0:
        return np.nan, 0, np.nan
    latest = weights.iloc[-1]
    if start_weight == target_weight:
        progress = 0
    else:
        progress = (start_weight - latest) / (start_weight - target_weight)
    progress = max(0, min(1, progress))
    return latest, progress, start_weight - latest

def estimate_target_date(df, target_weight):
    df = normalize_df(df)
    w = df.dropna(subset=["weight_kg"]).copy()
    if len(w) < 5:
        return None, None
    w["date_num"] = pd.to_datetime(w["date"]).map(pd.Timestamp.toordinal)
    recent = w.tail(min(len(w), 30))
    x = recent["date_num"].values
    y = recent["weight_kg"].values
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None, slope * 7
    target_num = (target_weight - intercept) / slope
    try:
        target_dt = datetime.fromordinal(int(target_num)).date()
        return target_dt, slope * 7
    except Exception:
        return None, slope * 7

# ============================================================
# Calendar utilities
# ============================================================

def build_month_calendar(df, year, month):
    df = normalize_df(df)
    data_by_date = {r["date"]: r for _, r in df.iterrows()}

    cal = calendar.Calendar(firstweekday=0)  # Monday
    weeks = cal.monthdatescalendar(year, month)

    rows = []
    for week in weeks:
        row = {}
        for day in week:
            is_current_month = day.month == month
            record = data_by_date.get(day)
            if not is_current_month:
                text = ""
            elif record is not None and bool(record.get("ran")):
                km = record.get("distance_km")
                text = f"🟩 {day.day}\n{km:.1f} km" if pd.notna(km) and km > 0 else f"🟩 {day.day}"
            elif record is not None:
                text = f"⬜ {day.day}\nNo run"
            else:
                text = f"▫️ {day.day}"
            row[day.strftime("%a")] = text
        rows.append(row)

    return pd.DataFrame(rows, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

def build_recent_activity(df, days=28):
    df = normalize_df(df)
    data_by_date = {r["date"]: r for _, r in df.iterrows()}
    end = date.today()
    start = end - timedelta(days=days - 1)

    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        record = data_by_date.get(d)
        if record is not None and bool(record.get("ran")):
            km = record.get("distance_km")
            status = "Run"
            distance = f"{km:.2f} km" if pd.notna(km) else "-"
            mark = "🟩"
        elif record is not None:
            status = "No run"
            distance = "-"
            mark = "⬜"
        else:
            status = "No data"
            distance = "-"
            mark = "▫️"

        rows.append({
            "Date": d.strftime("%d %b"),
            "Day": d.strftime("%a"),
            "Status": f"{mark} {status}",
            "Distance": distance
        })

    return pd.DataFrame(rows)

# ============================================================
# Apple Health parser
# ============================================================

def parse_duration_seconds(workout):
    duration = workout.attrib.get("duration")
    unit = workout.attrib.get("durationUnit", "min")
    if not duration:
        return np.nan
    duration = float(duration)
    unit = unit.lower()
    if unit in ["min", "minute", "minutes"]:
        return duration * 60
    if unit in ["s", "sec", "second", "seconds"]:
        return duration
    if unit in ["hr", "hour", "hours"]:
        return duration * 3600
    return duration * 60

def parse_apple_health(file):
    raw = file.read()
    if file.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            candidates = [name for name in z.namelist() if name.endswith("export.xml")]
            if not candidates:
                raise ValueError("export.xml was not found inside the Apple Health ZIP file.")
            xml_bytes = z.read(candidates[0])
    elif file.name.lower().endswith(".xml"):
        xml_bytes = raw
    else:
        raise ValueError("Please upload Apple Health export.zip or export.xml.")

    workouts = []
    weights = []
    context = ET.iterparse(io.BytesIO(xml_bytes), events=("end",))

    for event, elem in context:
        if elem.tag == "Workout":
            activity = elem.attrib.get("workoutActivityType", "")
            if "Running" in activity:
                start = elem.attrib.get("startDate")
                d = pd.to_datetime(start).date() if start else None
                duration_sec = parse_duration_seconds(elem)
                duration_min = duration_sec / 60 if not pd.isna(duration_sec) else np.nan
                distance_km = np.nan
                calories = np.nan

                for child in elem:
                    if child.tag == "WorkoutStatistics":
                        stat_type = child.attrib.get("type", "")
                        value = child.attrib.get("sum")
                        unit = child.attrib.get("unit", "")
                        if value is None:
                            continue
                        value = float(value)

                        if stat_type == "HKQuantityTypeIdentifierDistanceWalkingRunning":
                            if unit == "km":
                                distance_km = value
                            elif unit == "m":
                                distance_km = value / 1000
                            elif unit == "mi":
                                distance_km = value * 1.60934

                        if stat_type == "HKQuantityTypeIdentifierActiveEnergyBurned":
                            calories = value

                pace = calculate_pace(distance_km, duration_min)
                workouts.append({
                    "date": d,
                    "ran": True,
                    "distance_km": round(distance_km, 3) if pd.notna(distance_km) else np.nan,
                    "duration_min": round(duration_min, 2) if pd.notna(duration_min) else np.nan,
                    "pace_min_per_km": round(pace, 3) if pd.notna(pace) else np.nan,
                    "weight_kg": np.nan,
                    "calories_kcal": round(calories, 1) if pd.notna(calories) else np.nan,
                    "source": "Apple Health",
                    "notes": "Imported running workout"
                })

        elif elem.tag == "Record":
            rec_type = elem.attrib.get("type", "")
            start = elem.attrib.get("startDate")
            d = pd.to_datetime(start).date() if start else None
            value = elem.attrib.get("value")
            unit = elem.attrib.get("unit", "")

            if rec_type == "HKQuantityTypeIdentifierBodyMass" and value:
                weight = float(value)
                if unit.lower() in ["lb", "lbs"]:
                    weight *= 0.453592
                weights.append({
                    "date": d,
                    "ran": False,
                    "distance_km": np.nan,
                    "duration_min": np.nan,
                    "pace_min_per_km": np.nan,
                    "weight_kg": round(weight, 2),
                    "calories_kcal": np.nan,
                    "source": "Apple Health",
                    "notes": "Imported body weight"
                })

        elem.clear()

    workout_df = pd.DataFrame(workouts)
    weight_df = pd.DataFrame(weights)

    if not workout_df.empty:
        workout_df = workout_df.groupby("date", as_index=False).agg({
            "ran": "max",
            "distance_km": "sum",
            "duration_min": "sum",
            "calories_kcal": "sum",
            "source": "last",
            "notes": "last"
        })
        workout_df["pace_min_per_km"] = workout_df.apply(lambda r: calculate_pace(r["distance_km"], r["duration_min"]), axis=1)
        workout_df["weight_kg"] = np.nan

    if not weight_df.empty:
        weight_df = weight_df.groupby("date", as_index=False).agg({
            "ran": "last",
            "distance_km": "last",
            "duration_min": "last",
            "pace_min_per_km": "last",
            "weight_kg": "last",
            "calories_kcal": "last",
            "source": "last",
            "notes": "last"
        })

    imported = pd.concat([workout_df, weight_df], ignore_index=True)
    return normalize_df(imported)

# ============================================================
# UI styling
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
.app-title {
    font-size: 2.8rem;
    font-weight: 850;
    line-height: 1.15;
    margin-bottom: 0.35rem;
}
.app-subtitle {
    font-size: 1.08rem;
    color: #8A8F98;
    line-height: 1.55;
    margin-bottom: 1.35rem;
    max-width: 960px;
}
.section-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
}
.chart-title {
    font-size: 1.45rem;
    font-weight: 780;
    margin-top: 1.1rem;
    margin-bottom: 0.3rem;
}
.info-box {
    padding: 0.85rem 1rem;
    border-radius: 0.8rem;
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.22);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">🏃 Running & Weight Tracker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Track your runs, body weight, pace, estimated calories, training streaks, Apple Health imports, '
    'and weight-loss progress in one simple dashboard.'
    '</div>',
    unsafe_allow_html=True
)

df = load_data()

with st.sidebar:
    st.header("⚙️ Targets")
    start_weight = st.number_input("Starting weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
    target_weight = st.number_input("Target weight (kg)", min_value=40.0, max_value=150.0, value=60.0, step=0.5)
    weekly_km_target = st.number_input("Weekly running target (km)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

    current_weight_for_calorie = latest_weight_or_start(df, start_weight)
    weekly_calorie_estimate = estimate_calories(weekly_km_target, current_weight_for_calorie, True)

    st.divider()
    st.caption("Calorie target is calculated automatically from your weekly distance target.")
    st.metric("Estimated weekly calorie target", f"{weekly_calorie_estimate:.0f} kcal")
    st.caption("Formula: body weight × running distance. Apple Health calories are used when available.")

tabs = st.tabs([
    "🏠 Home",
    "➕ Input",
    "🍎 Apple Health",
    "📊 Analytics",
    "📅 Calendar",
    "💾 Data"
])

# ============================================================
# Home
# ============================================================

with tabs[0]:
    df = fill_missing_calories(df, start_weight)
    save_data(df)

    month_df = monthly_filter(df)
    week_df = weekly_filter(df)

    latest_weight, weight_progress, weight_lost = get_weight_progress(df, start_weight, target_weight)
    target_date, weekly_weight_rate = estimate_target_date(df, target_weight)

    total_km_month = month_df["distance_km"].fillna(0).sum() if not month_df.empty else 0
    total_cal_month = month_df["calories_kcal"].fillna(0).sum() if not month_df.empty else 0
    total_runs_month = int((month_df["ran"] == True).sum()) if not month_df.empty else 0
    avg_pace_month = month_df.loc[month_df["ran"] == True, "pace_min_per_km"].dropna().mean() if not month_df.empty else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Distance this month", f"{total_km_month:.1f} km")
    c2.metric("Calories this month", f"{total_cal_month:.0f} kcal")
    c3.metric("Runs this month", f"{total_runs_month}")
    c4.metric("Average pace", pace_to_text(avg_pace_month))
    c5.metric("Current streak", f"{current_streak(df)} days")

    st.divider()

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.markdown('<div class="section-title">Weight-Loss Progress</div>', unsafe_allow_html=True)
        if pd.isna(latest_weight):
            st.info("No body-weight data yet. Add your weight in the Input tab or import it from Apple Health.")
        else:
            st.metric("Latest weight", f"{latest_weight:.1f} kg", delta=f"{-weight_lost:.1f} kg from start")
            st.progress(weight_progress)
            st.caption(f"Target: {target_weight:.1f} kg | Progress: {weight_progress*100:.1f}%")
            if target_date:
                st.success(f"Estimated target date: **{target_date.strftime('%d %B %Y')}**")
                st.caption(f"Current trend: {weekly_weight_rate:.2f} kg/week")
            else:
                st.warning("Target-date prediction needs at least 5 body-weight entries with a clear downward trend.")

    with col_b:
        st.markdown('<div class="section-title">This Week</div>', unsafe_allow_html=True)
        week_km = week_df["distance_km"].fillna(0).sum() if not week_df.empty else 0
        week_cal = week_df["calories_kcal"].fillna(0).sum() if not week_df.empty else 0

        st.write("Running distance")
        st.progress(min(1, week_km / weekly_km_target) if weekly_km_target else 0)
        st.caption(f"{week_km:.1f} / {weekly_km_target:.1f} km")

        st.write("Estimated calories")
        st.progress(min(1, week_cal / weekly_calorie_estimate) if weekly_calorie_estimate else 0)
        st.caption(f"{week_cal:.0f} / {weekly_calorie_estimate:.0f} kcal")

    st.divider()

    st.markdown('<div class="section-title">Quick Summary</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No data yet. Start from the Input tab or import Apple Health data.")
    else:
        run_df = df[df["ran"] == True].copy()
        best_pace = run_df["pace_min_per_km"].dropna().min() if not run_df.empty else np.nan
        longest_run = run_df["distance_km"].dropna().max() if not run_df.empty else np.nan
        total_distance = run_df["distance_km"].dropna().sum() if not run_df.empty else 0
        total_runs = len(run_df)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total runs", f"{total_runs}")
        s2.metric("Total distance", f"{total_distance:.1f} km")
        s3.metric("Longest run", f"{longest_run:.1f} km" if pd.notna(longest_run) else "-")
        s4.metric("Best pace", pace_to_text(best_pace))

# ============================================================
# Input
# ============================================================

with tabs[1]:
    st.markdown('<div class="section-title">Daily Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Add one entry per day. Pace and calories are calculated automatically. '
        'If you do not enter calories manually, the app estimates them from your weight and running distance.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("daily_input", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_date = st.date_input("Date", value=date.today())
            ran = st.checkbox("Did you run today?", value=True)
        with c2:
            distance_km = st.number_input("Distance (km)", min_value=0.0, value=4.17, step=0.01)
            duration_text = st.text_input("Duration", value="32:03", help="Use MM:SS, HH:MM:SS, or total minutes.")
        with c3:
            weight_kg = st.number_input("Body weight today (kg)", min_value=0.0, value=float(current_weight_for_calorie), step=0.1)
            notes = st.text_input("Notes", value="")

        submitted = st.form_submit_button("💾 Save Entry")

    if submitted:
        try:
            duration_min = parse_duration_input(duration_text)
            pace = calculate_pace(distance_km, duration_min)
            calorie_weight = weight_kg if weight_kg > 0 else current_weight_for_calorie
            calories = estimate_calories(distance_km, calorie_weight, ran)

            new = pd.DataFrame([{
                "date": input_date,
                "ran": ran,
                "distance_km": distance_km if ran else 0,
                "duration_min": duration_min if ran else 0,
                "pace_min_per_km": pace if ran else np.nan,
                "weight_kg": weight_kg if weight_kg > 0 else np.nan,
                "calories_kcal": calories if ran else 0,
                "source": "Manual",
                "notes": notes
            }])
            df = upsert_rows(df, new)
            save_data(df)
            st.success(f"Saved ✅ Pace: {pace_to_text(pace)} | Estimated calories: {calories:.0f} kcal")
        except Exception as e:
            st.error(f"Could not save this entry: {e}")

# ============================================================
# Apple Health
# ============================================================

with tabs[2]:
    st.markdown('<div class="section-title">Apple Health Import</div>', unsafe_allow_html=True)

    st.markdown("""
    **How to export your Apple Health data from iPhone:**

    1. Open the **Health** app.
    2. Tap your profile picture in the top-right corner.
    3. Select **Export All Health Data**.
    4. Save the generated **export.zip** file.
    5. Upload the ZIP file here.

    The app will try to read running workouts, distance, duration, active calories, and body weight.
    """)

    uploaded = st.file_uploader("Upload Apple Health export.zip or export.xml", type=["zip", "xml"])

    if uploaded:
        try:
            imported = parse_apple_health(uploaded)
            if imported.empty:
                st.warning("No running workout or body-weight data was found.")
            else:
                st.success(f"Successfully read {len(imported)} rows from Apple Health.")
                st.dataframe(imported.sort_values("date", ascending=False), use_container_width=True)

                if st.button("Merge Apple Health Data"):
                    df = upsert_rows(df, imported)
                    df = fill_missing_calories(df, start_weight)
                    save_data(df)
                    st.success("Apple Health data has been merged ✅")
        except Exception as e:
            st.error(f"Could not read Apple Health data: {e}")

# ============================================================
# Analytics
# ============================================================

with tabs[3]:
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data yet.")
    else:
        dfp = normalize_df(df).sort_values("date")
        dfp["date"] = pd.to_datetime(dfp["date"])
        run_df = dfp[dfp["ran"] == True].copy()

        st.markdown('<div class="chart-title">Body Weight Over Time</div>', unsafe_allow_html=True)
        st.line_chart(dfp.set_index("date")[["weight_kg"]])
        st.caption("Daily body-weight entries.")

        if not dfp["weight_kg"].dropna().empty:
            tmp = dfp[["date", "weight_kg"]].dropna().copy()
            tmp["7-day moving average"] = tmp["weight_kg"].rolling(7, min_periods=1).mean()

            st.markdown('<div class="chart-title">Body Weight Trend</div>', unsafe_allow_html=True)
            st.line_chart(tmp.set_index("date")[["weight_kg", "7-day moving average"]])
            st.caption("The moving average helps reduce daily fluctuation noise.")

        st.markdown('<div class="chart-title">Running Distance</div>', unsafe_allow_html=True)
        st.line_chart(run_df.set_index("date")[["distance_km"]])
        st.caption("Distance per running session.")

        st.markdown('<div class="chart-title">Running Pace</div>', unsafe_allow_html=True)
        st.line_chart(run_df.set_index("date")[["pace_min_per_km"]])
        st.caption("Lower pace means faster running.")

        st.markdown('<div class="chart-title">Calories Burned</div>', unsafe_allow_html=True)
        st.line_chart(run_df.set_index("date")[["calories_kcal"]])
        st.caption("Apple Health calories are used when available. Otherwise, calories are estimated automatically.")

        if not run_df.empty:
            weekly = run_df.copy()
            weekly["week"] = weekly["date"].dt.to_period("W").astype(str)
            weekly_sum = weekly.groupby("week", as_index=False).agg(
                distance_km=("distance_km", "sum"),
                calories_kcal=("calories_kcal", "sum"),
                runs=("ran", "sum")
            )

            st.markdown('<div class="chart-title">Weekly Running Distance</div>', unsafe_allow_html=True)
            st.bar_chart(weekly_sum.set_index("week")[["distance_km"]])
            st.caption("Total running distance per week.")

        st.markdown('<div class="section-title">Personal Records</div>', unsafe_allow_html=True)
        if run_df.empty:
            st.info("No running data yet.")
        else:
            best_pace_row = run_df.loc[run_df["pace_min_per_km"].idxmin()] if run_df["pace_min_per_km"].dropna().size else None
            longest_row = run_df.loc[run_df["distance_km"].idxmax()] if run_df["distance_km"].dropna().size else None
            max_cal_row = run_df.loc[run_df["calories_kcal"].idxmax()] if run_df["calories_kcal"].dropna().size else None

            pr1, pr2, pr3 = st.columns(3)
            if best_pace_row is not None:
                pr1.metric("Best pace", pace_to_text(best_pace_row["pace_min_per_km"]))
                pr1.caption(best_pace_row["date"].strftime("%d %b %Y"))
            if longest_row is not None:
                pr2.metric("Longest run", f"{longest_row['distance_km']:.2f} km")
                pr2.caption(longest_row["date"].strftime("%d %b %Y"))
            if max_cal_row is not None:
                pr3.metric("Most calories", f"{max_cal_row['calories_kcal']:.0f} kcal")
                pr3.caption(max_cal_row["date"].strftime("%d %b %Y"))

# ============================================================
# Calendar
# ============================================================

with tabs[4]:
    st.markdown('<div class="section-title">Consistency Calendar</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data yet.")
    else:
        today = date.today()
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                index=today.month - 1,
                format_func=lambda m: calendar.month_name[m],
            )

        with col2:
            selected_year = st.selectbox(
                "Year",
                options=list(range(today.year - 2, today.year + 2)),
                index=2,
            )

        month_table = build_month_calendar(df, selected_year, selected_month)
        st.dataframe(month_table, use_container_width=True, hide_index=True)
        st.caption("🟩 Run recorded | ⬜ No run recorded | ▫️ No data")

        st.markdown('<div class="chart-title">Last 28 Days</div>', unsafe_allow_html=True)
        recent = build_recent_activity(df, 28)
        st.dataframe(recent.sort_index(ascending=False), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Longest streak", f"{longest_streak(df)} days")
        c2.metric("Current streak", f"{current_streak(df)} days")

# ============================================================
# Data
# ============================================================

with tabs[5]:
    st.markdown('<div class="section-title">Saved Data</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data yet.")
    else:
        display_df = normalize_df(df).sort_values("date", ascending=False).copy()
        display_df["pace"] = display_df["pace_min_per_km"].apply(pace_to_text)
        display_df["duration"] = display_df["duration_min"].apply(duration_to_text)

        ordered_columns = [
            "date", "ran", "distance_km", "duration", "pace",
            "weight_kg", "calories_kcal", "source", "notes"
        ]
        st.dataframe(display_df[ordered_columns], use_container_width=True)

        csv = normalize_df(df).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Backup CSV",
            data=csv,
            file_name="running_tracker_backup.csv",
            mime="text/csv"
        )

    st.markdown('<div class="section-title">Restore From CSV</div>', unsafe_allow_html=True)
    backup = st.file_uploader("Upload backup CSV", type=["csv"])
    if backup is not None:
        try:
            restored = pd.read_csv(backup)
            restored["source"] = restored.get("source", "CSV Restore")
            df = upsert_rows(df, restored)
            df = fill_missing_calories(df, start_weight)
            save_data(df)
            st.success("Backup data has been merged ✅")
        except Exception as e:
            st.error(f"Could not restore backup: {e}")

    st.warning("Delete all data only if you are sure.")
    if st.button("🗑️ Delete All Data"):
        if DATA_FILE.exists():
            DATA_FILE.unlink()
        st.success("All local data has been deleted. Refresh the page to restart.")
