
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, datetime, timedelta
import io
import math

DATA_FILE = Path("running_tracker_data.csv")

st.set_page_config(
    page_title="Akbar Running Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Utility functions
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
        raise ValueError("Format waktu harus MM:SS atau HH:MM:SS.")
    return float(text)

def estimate_calories(distance_km, weight_kg, ran=True):
    """
    Running estimate: about 1 kcal per kg per km.
    Walking/low intensity fallback: about 0.7 kcal per kg per km.
    """
    if pd.isna(distance_km) or pd.isna(weight_kg) or distance_km <= 0 or weight_kg <= 0:
        return np.nan
    factor = 1.0 if ran else 0.7
    return round(distance_km * weight_kg * factor, 1)

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

def fill_missing_calories(df, default_weight):
    df = normalize_df(df).sort_values("date").reset_index(drop=True)
    for idx, row in df.iterrows():
        if row["ran"] and (pd.isna(row["calories_kcal"]) or row["calories_kcal"] <= 0):
            weight = row["weight_kg"]
            if pd.isna(weight) or weight <= 0:
                prev = df.loc[:idx, "weight_kg"].dropna()
                weight = prev.iloc[-1] if len(prev) else default_weight
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

def make_calendar(df, days=84):
    end = date.today()
    start = end - timedelta(days=days - 1)
    all_dates = pd.date_range(start, end).date
    cal = pd.DataFrame({"date": all_dates})
    temp = normalize_df(df)
    cal = cal.merge(temp[["date", "ran", "distance_km"]], on="date", how="left")
    cal["weekday"] = pd.to_datetime(cal["date"]).dt.day_name().str[:3]
    cal["week"] = ((pd.to_datetime(cal["date"]) - pd.to_datetime(cal["date"]).min()).dt.days // 7)
    cal["mark"] = np.where(cal["ran"] == True, "🟩", "⬜")
    cal["text"] = cal.apply(
        lambda r: f"{r['mark']} {r['date'].strftime('%d/%m')} ({r['distance_km']:.1f} km)"
        if pd.notna(r["distance_km"]) and r["distance_km"] > 0 else f"{r['mark']} {r['date'].strftime('%d/%m')}",
        axis=1,
    )
    return cal

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
# Apple Health parsing
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
                raise ValueError("Tidak menemukan export.xml di dalam ZIP Apple Health.")
            xml_bytes = z.read(candidates[0])
    elif file.name.lower().endswith(".xml"):
        xml_bytes = raw
    else:
        raise ValueError("Upload file Apple Health export.zip atau export.xml.")

    workouts = []
    weights = []
    vo2max = []
    heart_rates = []

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
# App UI
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.big-title {font-size: 2.6rem; font-weight: 800; margin-bottom: 0.1rem;}
.subtitle {font-size: 1.05rem; color: #888; margin-bottom: 1.2rem;}
.metric-card {
    padding: 1.0rem;
    border-radius: 1rem;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.08);
}
.mobile-note {
    padding: 0.8rem 1rem;
    border-radius: 0.8rem;
    background: rgba(255, 165, 0, 0.10);
    border: 1px solid rgba(255, 165, 0, 0.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🏃 Akbar Running & Weight Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tracking lari, berat badan, pace, kalori, streak, Apple Health import, dan prediksi target berat badan.</div>', unsafe_allow_html=True)

df = load_data()

with st.sidebar:
    st.header("⚙️ Target & Settings")
    start_weight = st.number_input("Berat awal program (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
    target_weight = st.number_input("Target berat badan (kg)", min_value=40.0, max_value=150.0, value=60.0, step=0.5)
    weekly_km_target = st.number_input("Target lari mingguan (km)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    calorie_target = st.number_input("Target kalori lari mingguan (kcal)", min_value=0, max_value=10000, value=700, step=50)
    default_weight = st.number_input("Berat default untuk estimasi kalori (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
    st.caption("Kalori dihitung dari Apple Health jika tersedia. Jika tidak, estimasi: ±1 kcal × berat badan × km.")

tabs = st.tabs([
    "🏠 Home",
    "➕ Input",
    "🍎 Apple Health",
    "📊 Analytics",
    "📅 Calendar",
    "💾 Data"
])

# ---------------- Home ----------------

with tabs[0]:
    df = fill_missing_calories(df, default_weight)
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
    c1.metric("Km bulan ini", f"{total_km_month:.1f} km")
    c2.metric("Kalori bulan ini", f"{total_cal_month:.0f} kcal")
    c3.metric("Run bulan ini", f"{total_runs_month}x")
    c4.metric("Avg pace", pace_to_text(avg_pace_month))
    c5.metric("Streak", f"{current_streak(df)} hari")

    st.divider()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("Progress berat badan")
        if pd.isna(latest_weight):
            st.info("Belum ada data berat badan.")
        else:
            st.metric("Berat terakhir", f"{latest_weight:.1f} kg", delta=f"{-weight_lost:.1f} kg dari awal")
            st.progress(weight_progress)
            st.caption(f"Target: {target_weight:.1f} kg | Progress: {weight_progress*100:.1f}%")
            if target_date:
                st.success(f"Estimasi mencapai target: **{target_date.strftime('%d %B %Y')}**")
                st.caption(f"Tren berat: {weekly_weight_rate:.2f} kg/minggu")
            else:
                st.warning("Prediksi target belum tersedia. Butuh minimal ±5 data berat badan dengan tren menurun.")

    with col_b:
        st.subheader("Target minggu ini")
        week_km = week_df["distance_km"].fillna(0).sum() if not week_df.empty else 0
        week_cal = week_df["calories_kcal"].fillna(0).sum() if not week_df.empty else 0

        st.write("Jarak")
        st.progress(min(1, week_km / weekly_km_target) if weekly_km_target else 0)
        st.caption(f"{week_km:.1f} / {weekly_km_target:.1f} km")

        st.write("Kalori")
        st.progress(min(1, week_cal / calorie_target) if calorie_target else 0)
        st.caption(f"{week_cal:.0f} / {calorie_target:.0f} kcal")

    st.divider()

    st.subheader("Ringkasan cepat")
    if df.empty:
        st.info("Belum ada data. Mulai dari tab Input atau import Apple Health.")
    else:
        run_df = df[df["ran"] == True].copy()
        best_pace = run_df["pace_min_per_km"].dropna().min() if not run_df.empty else np.nan
        longest_run = run_df["distance_km"].dropna().max() if not run_df.empty else np.nan
        total_distance = run_df["distance_km"].dropna().sum() if not run_df.empty else 0
        total_runs = len(run_df)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total semua run", f"{total_runs}x")
        s2.metric("Total semua jarak", f"{total_distance:.1f} km")
        s3.metric("Longest run", f"{longest_run:.1f} km" if pd.notna(longest_run) else "-")
        s4.metric("Best pace", pace_to_text(best_pace))

# ---------------- Input ----------------

with tabs[1]:
    st.subheader("Input harian")
    st.markdown('<div class="mobile-note">Tips iPhone: setelah app dibuka di Safari, pilih Share → Add to Home Screen supaya terasa seperti aplikasi.</div>', unsafe_allow_html=True)

    with st.form("daily_input", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_date = st.date_input("Tanggal", value=date.today())
            ran = st.checkbox("Hari ini lari?", value=True)
        with c2:
            distance_km = st.number_input("Jarak lari (km)", min_value=0.0, value=4.17, step=0.01)
            duration_text = st.text_input("Waktu lari", value="32:03", help="Format: MM:SS, HH:MM:SS, atau total menit.")
        with c3:
            weight_kg = st.number_input("Berat badan hari ini (kg)", min_value=0.0, value=default_weight, step=0.1)
            notes = st.text_input("Catatan", value="")

        submitted = st.form_submit_button("💾 Simpan data")

    if submitted:
        try:
            duration_min = parse_duration_input(duration_text)
            pace = calculate_pace(distance_km, duration_min)
            calories = estimate_calories(distance_km, weight_kg, ran)

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
            st.success(f"Tersimpan ✅ Pace: {pace_to_text(pace)} | Kalori: {calories:.0f} kcal")
        except Exception as e:
            st.error(f"Gagal menyimpan: {e}")

# ---------------- Apple Health ----------------

with tabs[2]:
    st.subheader("Import Apple Health")

    st.markdown("""
    **Cara export dari iPhone:**
    1. Buka aplikasi **Health / Kesehatan**
    2. Tap foto profil kanan atas
    3. Pilih **Export All Health Data**
    4. Simpan file **export.zip**
    5. Upload di sini

    Data yang dibaca:
    - Running workout
    - Jarak dan durasi
    - Active calories jika tersedia
    - Berat badan jika tercatat di Apple Health
    """)

    uploaded = st.file_uploader("Upload Apple Health export.zip atau export.xml", type=["zip", "xml"])

    if uploaded:
        try:
            imported = parse_apple_health(uploaded)
            if imported.empty:
                st.warning("Tidak ada running workout atau berat badan yang terbaca.")
            else:
                st.success(f"Berhasil membaca {len(imported)} baris data.")
                st.dataframe(imported.sort_values("date", ascending=False), use_container_width=True)

                if st.button("Gabungkan ke tracker"):
                    df = upsert_rows(df, imported)
                    df = fill_missing_calories(df, default_weight)
                    save_data(df)
                    st.success("Data Apple Health berhasil digabungkan ✅")
        except Exception as e:
            st.error(f"Gagal membaca file Apple Health: {e}")

# ---------------- Analytics ----------------

with tabs[3]:
    st.subheader("Analytics")

    if df.empty:
        st.info("Belum ada data.")
    else:
        dfp = normalize_df(df).sort_values("date")
        dfp["date"] = pd.to_datetime(dfp["date"])
        run_df = dfp[dfp["ran"] == True].copy()

        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(dfp.set_index("date")[["weight_kg"]])
            st.caption("Berat badan")
        with c2:
            if not dfp["weight_kg"].dropna().empty:
                tmp = dfp[["date", "weight_kg"]].dropna().copy()
                tmp["weight_7d_avg"] = tmp["weight_kg"].rolling(7, min_periods=1).mean()
                st.line_chart(tmp.set_index("date")[["weight_kg", "weight_7d_avg"]])
                st.caption("Berat badan + trendline 7 hari")

        c3, c4 = st.columns(2)
        with c3:
            st.line_chart(run_df.set_index("date")[["distance_km"]])
            st.caption("Jarak lari")
        with c4:
            st.line_chart(run_df.set_index("date")[["pace_min_per_km"]])
            st.caption("Pace min/km — semakin rendah semakin cepat")

        c5, c6 = st.columns(2)
        with c5:
            st.line_chart(run_df.set_index("date")[["calories_kcal"]])
            st.caption("Kalori terbakar")
        with c6:
            weekly = run_df.copy()
            weekly["week"] = weekly["date"].dt.to_period("W").astype(str)
            weekly_sum = weekly.groupby("week", as_index=False).agg(
                distance_km=("distance_km", "sum"),
                calories_kcal=("calories_kcal", "sum"),
                runs=("ran", "sum")
            )
            st.bar_chart(weekly_sum.set_index("week")[["distance_km"]])
            st.caption("Total km per minggu")

        st.subheader("Personal records")
        if run_df.empty:
            st.info("Belum ada data lari.")
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

# ---------------- Calendar ----------------

with tabs[4]:
    st.subheader("Kalender konsistensi")

    if df.empty:
        st.info("Belum ada data.")
    else:
        cal = make_calendar(df, 84)
        pivot = cal.pivot(index="week", columns="weekday", values="text")
        desired = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = pivot.reindex(columns=[d for d in desired if d in pivot.columns])
        st.dataframe(pivot, use_container_width=True)
        st.caption("🟩 = lari, ⬜ = tidak ada data/tidak lari")

        st.metric("Longest streak", f"{longest_streak(df)} hari")
        st.metric("Current streak", f"{current_streak(df)} hari")

# ---------------- Data ----------------

with tabs[5]:
    st.subheader("Data tersimpan")

    if df.empty:
        st.info("Belum ada data.")
    else:
        display_df = normalize_df(df).sort_values("date", ascending=False).copy()
        display_df["pace_text"] = display_df["pace_min_per_km"].apply(pace_to_text)
        display_df["duration_text"] = display_df["duration_min"].apply(duration_to_text)
        st.dataframe(display_df, use_container_width=True)

        csv = normalize_df(df).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download backup CSV",
            data=csv,
            file_name="running_tracker_backup.csv",
            mime="text/csv"
        )

    st.subheader("Restore dari CSV")
    backup = st.file_uploader("Upload backup CSV", type=["csv"])
    if backup is not None:
        try:
            restored = pd.read_csv(backup)
            restored["source"] = restored.get("source", "CSV Restore")
            df = upsert_rows(df, restored)
            df = fill_missing_calories(df, default_weight)
            save_data(df)
            st.success("Backup berhasil digabungkan ✅")
        except Exception as e:
            st.error(f"Gagal restore CSV: {e}")

    st.warning("Hapus semua data hanya jika benar-benar perlu.")
    if st.button("🗑️ Hapus semua data"):
        if DATA_FILE.exists():
            DATA_FILE.unlink()
        st.success("Semua data lokal dihapus. Refresh halaman untuk mulai ulang.")
