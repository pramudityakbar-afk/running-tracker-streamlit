
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
    page_title="Akbar Running & Weight Tracker",
    page_icon="🏃",
    layout="wide"
)

# ---------- Helpers ----------

def load_data():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=[
        "date", "ran", "distance_km", "duration_min", "pace_min_per_km",
        "weight_kg", "calories_kcal", "source", "notes"
    ])

def save_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    df.to_csv(DATA_FILE, index=False)

def pace_to_text(pace):
    if pd.isna(pace) or pace <= 0 or math.isinf(pace):
        return "-"
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} min/km"

def calculate_pace(distance_km, duration_min):
    if distance_km and distance_km > 0 and duration_min and duration_min > 0:
        return duration_min / distance_km
    return np.nan

def estimate_calories(distance_km, weight_kg, duration_min=None, ran=True):
    """
    Running rule of thumb: ~1 kcal per kg per km.
    Walking / low intensity fallback: ~0.7 kcal per kg per km.
    """
    if not distance_km or distance_km <= 0 or not weight_kg or weight_kg <= 0:
        return np.nan
    factor = 1.0 if ran else 0.7
    return round(distance_km * weight_kg * factor, 1)

def upsert_rows(existing, new_rows):
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    # Prefer Apple Health / imported rows if same date has run data,
    # otherwise keep latest non-empty values.
    combined["priority"] = combined["source"].map({
        "Apple Health": 3,
        "Manual": 2,
        "CSV Restore": 1
    }).fillna(0)

    combined = combined.sort_values(["date", "priority"])
    result_rows = []

    for d, group in combined.groupby("date"):
        final = {}
        final["date"] = d
        final["ran"] = bool(group["ran"].fillna(False).iloc[-1])
        for col in ["distance_km", "duration_min", "pace_min_per_km", "weight_kg", "calories_kcal", "source", "notes"]:
            values = group[col].replace("", np.nan).dropna()
            final[col] = values.iloc[-1] if len(values) else np.nan
        result_rows.append(final)

    result = pd.DataFrame(result_rows)
    if "priority" in result.columns:
        result = result.drop(columns=["priority"])
    return result.sort_values("date").reset_index(drop=True)

def parse_duration_seconds(workout):
    # Apple Health Workout has duration + durationUnit, commonly "min".
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
    """
    Accepts Apple Health export.zip or export.xml.
    Extracts:
    - HKWorkoutActivityTypeRunning workouts
    - HKQuantityTypeIdentifierBodyMass records
    - ActiveEnergyBurned when attached as WorkoutStatistics, if present
    """
    raw = file.read()
    xml_bytes = None

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

    # iterparse is memory-friendly
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
                            # Apple usually exports kcal as Cal/kcal depending locale.
                            calories = value

                pace = calculate_pace(distance_km, duration_min)

                workouts.append({
                    "date": d,
                    "ran": True,
                    "distance_km": round(distance_km, 3) if not pd.isna(distance_km) else np.nan,
                    "duration_min": round(duration_min, 2) if not pd.isna(duration_min) else np.nan,
                    "pace_min_per_km": round(pace, 3) if not pd.isna(pace) else np.nan,
                    "weight_kg": np.nan,
                    "calories_kcal": round(calories, 1) if not pd.isna(calories) else np.nan,
                    "source": "Apple Health",
                    "notes": "Imported running workout"
                })

        elif elem.tag == "Record":
            rec_type = elem.attrib.get("type", "")
            if rec_type == "HKQuantityTypeIdentifierBodyMass":
                start = elem.attrib.get("startDate")
                d = pd.to_datetime(start).date() if start else None
                value = elem.attrib.get("value")
                unit = elem.attrib.get("unit", "kg")
                if value:
                    weight = float(value)
                    if unit.lower() in ["lb", "lbs"]:
                        weight *= 0.453592
                    weights.append({
                        "date": d,
                        "ran": np.nan,
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
        workout_df["pace_min_per_km"] = workout_df.apply(
            lambda r: calculate_pace(r["distance_km"], r["duration_min"]), axis=1
        )
        workout_df["pace_min_per_km"] = workout_df["pace_min_per_km"].round(3)
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
    if imported.empty:
        return imported

    imported["date"] = pd.to_datetime(imported["date"]).dt.date
    return imported

def make_calendar(df, days=90):
    end = date.today()
    start = end - timedelta(days=days-1)
    all_dates = pd.date_range(start, end).date
    cal = pd.DataFrame({"date": all_dates})
    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"]).dt.date
    cal = cal.merge(temp[["date", "ran", "distance_km"]], on="date", how="left")
    cal["status"] = np.where(cal["ran"] == True, "🟢", "⚪")
    cal["label"] = cal.apply(
        lambda r: f"{r['status']} {r['date'].strftime('%d/%m')} ({r['distance_km']:.2f} km)" 
        if pd.notna(r["distance_km"]) else f"{r['status']} {r['date'].strftime('%d/%m')}",
        axis=1
    )
    return cal

def current_streak(df):
    temp = df.copy()
    if temp.empty:
        return 0
    temp["date"] = pd.to_datetime(temp["date"]).dt.date
    ran_dates = set(temp.loc[temp["ran"] == True, "date"])
    d = date.today()
    streak = 0
    while d in ran_dates:
        streak += 1
        d -= timedelta(days=1)
    return streak

# ---------- UI ----------

st.title("🏃 Running, Weight & Apple Health Tracker")
st.caption("Track lari, berat badan, pace, kalori, streak, dan progress dari input manual atau Apple Health export.")

df = load_data()

with st.sidebar:
    st.header("⚙️ Target")
    target_weight = st.number_input("Target berat badan (kg)", min_value=40.0, max_value=150.0, value=60.0, step=0.5)
    weekly_km_target = st.number_input("Target lari mingguan (km)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    default_weight = st.number_input("Berat badan default untuk estimasi kalori (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)

tab1, tab2, tab3, tab4 = st.tabs([
    "➕ Input Harian",
    "🍎 Import Apple Health",
    "📊 Dashboard",
    "💾 Data"
])

with tab1:
    st.subheader("Input harian")

    with st.form("daily_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_date = st.date_input("Tanggal", value=date.today())
            ran = st.checkbox("Hari ini lari?", value=True)
        with c2:
            distance_km = st.number_input("Jarak lari (km)", min_value=0.0, value=4.17, step=0.01)
            duration_text = st.text_input("Waktu lari (MM:SS atau menit)", value="32:03")
        with c3:
            weight_kg = st.number_input("Berat badan hari ini (kg)", min_value=0.0, value=default_weight, step=0.1)
            notes = st.text_input("Catatan", value="")

        submitted = st.form_submit_button("Simpan")

    if submitted:
        try:
            if ":" in duration_text:
                parts = duration_text.split(":")
                duration_min = int(parts[0]) + int(parts[1]) / 60
            else:
                duration_min = float(duration_text)

            pace = calculate_pace(distance_km, duration_min)
            calories = estimate_calories(distance_km, weight_kg, duration_min, ran)

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
            st.success(f"Tersimpan. Pace: {pace_to_text(pace)} | Kalori: {calories} kcal")
        except Exception as e:
            st.error(f"Gagal menyimpan input: {e}")

with tab2:
    st.subheader("Import dari Apple Health")

    st.markdown("""
    **Cara export dari iPhone:**
    1. Buka aplikasi **Health / Kesehatan**
    2. Tap foto profil di kanan atas
    3. Pilih **Export All Health Data**
    4. Simpan atau kirim file **export.zip**
    5. Upload file ZIP tersebut di sini

    Aplikasi ini akan mencoba mengambil:
    - Running workout
    - Distance Walking/Running
    - Active energy burned
    - Body mass / berat badan
    """)

    uploaded = st.file_uploader("Upload Apple Health export.zip atau export.xml", type=["zip", "xml"])

    if uploaded is not None:
        try:
            imported = parse_apple_health(uploaded)

            if imported.empty:
                st.warning("Tidak ada data running workout atau berat badan yang terbaca dari Apple Health export.")
            else:
                st.success(f"Berhasil membaca {len(imported)} baris data dari Apple Health.")
                st.dataframe(imported.sort_values("date", ascending=False), use_container_width=True)

                if st.button("Gabungkan ke tracker"):
                    df = upsert_rows(df, imported)
                    # Fill missing calories using latest available weight/default
                    for idx, row in df.iterrows():
                        if row.get("ran") == True and (pd.isna(row.get("calories_kcal")) or row.get("calories_kcal") == 0):
                            weight = row.get("weight_kg")
                            if pd.isna(weight) or weight <= 0:
                                prev_weights = df.loc[df["date"] <= row["date"], "weight_kg"].dropna()
                                weight = prev_weights.iloc[-1] if len(prev_weights) else default_weight
                            df.at[idx, "calories_kcal"] = estimate_calories(row["distance_km"], weight, row["duration_min"], True)
                    save_data(df)
                    st.success("Data Apple Health berhasil digabungkan.")
        except Exception as e:
            st.error(f"Gagal membaca Apple Health export: {e}")

with tab3:
    st.subheader("Dashboard progress")

    if df.empty:
        st.info("Belum ada data. Isi input harian atau import Apple Health terlebih dahulu.")
    else:
        df_plot = df.copy()
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        df_plot = df_plot.sort_values("date")

        total_km = df_plot["distance_km"].fillna(0).sum()
        total_runs = int((df_plot["ran"] == True).sum())
        avg_pace = df_plot.loc[df_plot["ran"] == True, "pace_min_per_km"].dropna().mean()
        latest_weight = df_plot["weight_kg"].dropna().iloc[-1] if df_plot["weight_kg"].dropna().size else np.nan
        total_cal = df_plot["calories_kcal"].fillna(0).sum()
        streak = current_streak(df)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total lari", f"{total_runs}x")
        m2.metric("Total jarak", f"{total_km:.2f} km")
        m3.metric("Avg pace", pace_to_text(avg_pace))
        m4.metric("Kalori total", f"{total_cal:.0f} kcal")
        m5.metric("Current streak", f"{streak} hari")

        if not pd.isna(latest_weight):
            st.progress(max(0, min(1, (default_weight - latest_weight) / (default_weight - target_weight))) if default_weight != target_weight else 0)
            st.caption(f"Berat terakhir: {latest_weight:.1f} kg | Target: {target_weight:.1f} kg")

        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_plot.set_index("date")[["distance_km"]])
            st.caption("Jarak lari per hari")
        with c2:
            st.line_chart(df_plot.set_index("date")[["pace_min_per_km"]])
            st.caption("Pace min/km — semakin turun berarti semakin cepat")

        c3, c4 = st.columns(2)
        with c3:
            st.line_chart(df_plot.set_index("date")[["weight_kg"]])
            st.caption("Berat badan")
        with c4:
            st.line_chart(df_plot.set_index("date")[["calories_kcal"]])
            st.caption("Kalori terbakar")

        st.subheader("Kalender konsistensi 90 hari")
        cal = make_calendar(df, 90)
        weeks = []
        labels = cal["label"].tolist()
        for i in range(0, len(labels), 7):
            weeks.append(labels[i:i+7])
        st.dataframe(pd.DataFrame(weeks, columns=["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]), use_container_width=True)

        st.subheader("Target minggu ini")
        today = pd.Timestamp.today()
        week_start = today - pd.Timedelta(days=today.weekday())
        week_df = df_plot[df_plot["date"] >= week_start]
        week_km = week_df["distance_km"].fillna(0).sum()
        st.progress(min(1, week_km / weekly_km_target) if weekly_km_target else 0)
        st.caption(f"{week_km:.2f} km dari target {weekly_km_target:.2f} km")

with tab4:
    st.subheader("Data tersimpan")

    if df.empty:
        st.info("Belum ada data.")
    else:
        display_df = df.sort_values("date", ascending=False).copy()
        display_df["pace"] = display_df["pace_min_per_km"].apply(pace_to_text)
        st.dataframe(display_df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download backup CSV",
            data=csv,
            file_name="running_tracker_backup.csv",
            mime="text/csv"
        )

    st.subheader("Restore dari CSV")
    backup = st.file_uploader("Upload backup CSV", type=["csv"])
    if backup is not None:
        try:
            restored = pd.read_csv(backup)
            restored["date"] = pd.to_datetime(restored["date"]).dt.date
            df = upsert_rows(df, restored)
            save_data(df)
            st.success("Backup CSV berhasil digabungkan.")
        except Exception as e:
            st.error(f"Gagal restore CSV: {e}")

    st.warning("Hapus semua data hanya jika benar-benar perlu.")
    if st.button("Hapus semua data"):
        if DATA_FILE.exists():
            DATA_FILE.unlink()
        st.success("Semua data lokal telah dihapus. Refresh halaman untuk mulai ulang.")
