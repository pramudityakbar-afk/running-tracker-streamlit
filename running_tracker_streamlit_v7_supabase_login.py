
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import math
import calendar
from supabase import create_client, Client

st.set_page_config(
    page_title="Running & Weight Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Supabase connection
# ============================================================

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

def get_supabase() -> Client:
    client = init_supabase()
    session = st.session_state.get("session")

    # Attach the logged-in user's JWT so Supabase RLS can isolate the user's data.
    if session:
        try:
            client.auth.set_session(session["access_token"], session["refresh_token"])
        except Exception:
            pass

    return client

def get_user_id():
    user = st.session_state.get("user")
    if not user:
        return None
    return user.get("id")

# ============================================================
# General utilities
# ============================================================

REQUIRED_COLUMNS = [
    "id", "user_id", "date", "ran", "distance_km", "duration_min",
    "pace_min_per_km", "weight_kg", "calories_kcal", "source", "notes"
]

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
    if pd.isna(distance_km) or pd.isna(weight_kg) or distance_km <= 0 or weight_kg <= 0:
        return np.nan
    factor = 1.0 if ran else 0.7
    return round(distance_km * weight_kg * factor, 1)

def calculate_ideal_weight_range(height_cm: float, gender: str):
    if height_cm <= 0:
        return np.nan, np.nan, np.nan

    height_m = height_cm / 100
    healthy_min = 18.5 * height_m * height_m
    healthy_max = 24.9 * height_m * height_m

    height_in = height_cm / 2.54
    inches_over_5ft = max(0, height_in - 60)

    if gender == "Male":
        devine = 50 + 2.3 * inches_over_5ft
    elif gender == "Female":
        devine = 45.5 + 2.3 * inches_over_5ft
    else:
        devine = 22 * height_m * height_m

    return round(healthy_min, 1), round(healthy_max, 1), round(devine, 1)

def calculate_bmi(weight_kg: float, height_cm: float):
    if weight_kg <= 0 or height_cm <= 0:
        return np.nan
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 1)

def bmi_category(bmi: float):
    if pd.isna(bmi):
        return "-"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal range"
    if bmi < 30:
        return "Overweight"
    return "Obesity range"

def normalize_entries(rows):
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["ran"] = df["ran"].fillna(False).astype(bool)

    for col in ["distance_km", "duration_min", "pace_min_per_km", "weight_kg", "calories_kcal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source"] = df["source"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df.sort_values("date")

def latest_weight_or_start(df: pd.DataFrame, start_weight: float) -> float:
    if df.empty:
        return float(start_weight)
    weights = df["weight_kg"].dropna()
    if len(weights):
        return float(weights.iloc[-1])
    return float(start_weight)

def fill_missing_calories(df, start_weight):
    df = df.sort_values("date").reset_index(drop=True).copy()
    for idx, row in df.iterrows():
        if row["ran"] and (pd.isna(row["calories_kcal"]) or row["calories_kcal"] <= 0):
            weight = row["weight_kg"]
            if pd.isna(weight) or weight <= 0:
                prev = df.loc[:idx, "weight_kg"].dropna()
                weight = prev.iloc[-1] if len(prev) else start_weight
            df.at[idx, "calories_kcal"] = estimate_calories(row["distance_km"], weight, True)
    return df

def current_streak(df):
    if df.empty:
        return 0
    ran_dates = set(df.loc[df["ran"] == True, "date"])
    d = date.today()
    streak = 0
    while d in ran_dates:
        streak += 1
        d -= timedelta(days=1)
    return streak

def longest_streak(df):
    if df.empty:
        return 0
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
    if df.empty:
        return df
    today = date.today()
    return df[(pd.to_datetime(df["date"]).dt.year == today.year) & (pd.to_datetime(df["date"]).dt.month == today.month)]

def weekly_filter(df):
    if df.empty:
        return df
    today = pd.Timestamp.today().date()
    week_start = today - timedelta(days=today.weekday())
    return df[df["date"] >= week_start]

def get_weight_progress(df, start_weight, target_weight):
    weights = df["weight_kg"].dropna() if not df.empty else pd.Series(dtype=float)
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
    if df.empty:
        return None, None
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
# Supabase data functions
# ============================================================

def load_profile():
    user_id = get_user_id()
    if not user_id:
        return None

    supabase = get_supabase()
    result = supabase.table("user_profiles").select("*").eq("user_id", user_id).limit(1).execute()

    if result.data:
        return result.data[0]

    default_profile = {
        "user_id": user_id,
        "gender": "Male",
        "height_cm": 165,
        "start_weight_kg": 70,
        "target_weight_kg": 60,
        "weekly_km_target": 30,
        "use_ideal_as_target": True,
    }

    supabase.table("user_profiles").insert(default_profile).execute()
    return default_profile

def save_profile(profile):
    user_id = get_user_id()
    if not user_id:
        return

    profile["user_id"] = user_id
    supabase = get_supabase()
    supabase.table("user_profiles").upsert(profile, on_conflict="user_id").execute()

def load_entries():
    user_id = get_user_id()
    if not user_id:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    supabase = get_supabase()
    result = (
        supabase.table("running_entries")
        .select("*")
        .eq("user_id", user_id)
        .order("date")
        .execute()
    )

    return normalize_entries(result.data)

def save_entry(entry):
    user_id = get_user_id()
    if not user_id:
        return

    entry["user_id"] = user_id

    supabase = get_supabase()

    # One entry per user per date.
    existing = (
        supabase.table("running_entries")
        .select("id")
        .eq("user_id", user_id)
        .eq("date", entry["date"])
        .limit(1)
        .execute()
    )

    if existing.data:
        entry_id = existing.data[0]["id"]
        supabase.table("running_entries").update(entry).eq("id", entry_id).execute()
    else:
        supabase.table("running_entries").insert(entry).execute()

def delete_entry(entry_id):
    supabase = get_supabase()
    supabase.table("running_entries").delete().eq("id", entry_id).execute()

# ============================================================
# Calendar utilities
# ============================================================

def build_month_calendar(df, year, month):
    data_by_date = {r["date"]: r for _, r in df.iterrows()} if not df.empty else {}

    cal = calendar.Calendar(firstweekday=0)
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
                text = f"🟩 {day.day}  |  {km:.1f} km" if pd.notna(km) and km > 0 else f"🟩 {day.day}"
            elif record is not None:
                text = f"⬜ {day.day}  |  No run"
            else:
                text = f"▫️ {day.day}"
            row[day.strftime("%a")] = text
        rows.append(row)

    return pd.DataFrame(rows, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

def build_recent_activity(df, days=28):
    data_by_date = {r["date"]: r for _, r in df.iterrows()} if not df.empty else {}
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
# Authentication UI
# ============================================================

def auth_screen():
    st.markdown("""
    <style>
    .auth-title {
        font-size: 2.4rem;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }
    .auth-subtitle {
        font-size: 1.05rem;
        color: #8A8F98;
        margin-bottom: 1.5rem;
        max-width: 780px;
        line-height: 1.55;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-title">🏃 Running & Weight Tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-subtitle">'
        'Create an account or sign in to track your running progress, body weight, pace, calories, streaks, '
        'and ideal-weight target privately.'
        '</div>',
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                supabase = get_supabase()
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state["user"] = {
                    "id": res.user.id,
                    "email": res.user.email,
                }
                st.session_state["session"] = {
                    "access_token": res.session.access_token,
                    "refresh_token": res.session.refresh_token,
                }
                st.success("Logged in successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            try:
                supabase = get_supabase()
                res = supabase.auth.sign_up({"email": email, "password": password})

                if res.user and res.session:
                    st.session_state["user"] = {
                        "id": res.user.id,
                        "email": res.user.email,
                    }
                    st.session_state["session"] = {
                        "access_token": res.session.access_token,
                        "refresh_token": res.session.refresh_token,
                    }
                    st.success("Account created and logged in.")
                    st.rerun()
                else:
                    st.info("Account created. Please check your email to confirm your account before logging in.")
            except Exception as e:
                st.error(f"Sign-up failed: {e}")

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

if "user" not in st.session_state or "session" not in st.session_state:
    auth_screen()
    st.stop()

# ============================================================
# Main app
# ============================================================

st.markdown('<div class="app-title">🏃 Running & Weight Tracker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Track your runs, body weight, pace, estimated calories, training streaks, Apple Health imports, '
    'BMI, estimated ideal weight, and weight-loss progress in one simple dashboard.'
    '</div>',
    unsafe_allow_html=True
)

profile = load_profile()
df = load_entries()

with st.sidebar:
    st.header("👤 Account")
    st.write(st.session_state["user"]["email"])
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.header("👤 Profile")

    gender_options = ["Male", "Female", "Other / Prefer not to say"]
    gender = st.selectbox(
        "Gender",
        gender_options,
        index=gender_options.index(profile.get("gender", "Male")) if profile.get("gender", "Male") in gender_options else 0,
    )
    height_cm = st.number_input(
        "Height (cm)",
        min_value=120.0,
        max_value=230.0,
        value=float(profile.get("height_cm", 165)),
        step=1.0,
    )
    start_weight = st.number_input(
        "Current / starting weight (kg)",
        min_value=35.0,
        max_value=250.0,
        value=float(profile.get("start_weight_kg", 70)),
        step=0.5,
    )

    healthy_min, healthy_max, ideal_weight = calculate_ideal_weight_range(height_cm, gender)
    current_bmi = calculate_bmi(start_weight, height_cm)

    st.metric("BMI", f"{current_bmi:.1f}" if pd.notna(current_bmi) else "-")
    st.caption(f"Category: {bmi_category(current_bmi)}")
    st.metric("Estimated ideal weight", f"{ideal_weight:.1f} kg")
    st.caption(f"Healthy BMI range: {healthy_min:.1f}–{healthy_max:.1f} kg")

    st.divider()
    st.header("🎯 Targets")

    use_ideal_as_target = st.checkbox(
        "Use estimated ideal weight as target",
        value=bool(profile.get("use_ideal_as_target", True)),
    )

    if use_ideal_as_target:
        target_weight = ideal_weight
        st.info(f"Target weight is automatically set to {target_weight:.1f} kg.")
    else:
        target_weight = st.number_input(
            "Custom target weight (kg)",
            min_value=35.0,
            max_value=250.0,
            value=float(profile.get("target_weight_kg", ideal_weight)),
            step=0.5,
        )

    weekly_km_target = st.number_input(
        "Weekly running target (km)",
        min_value=0.0,
        max_value=100.0,
        value=float(profile.get("weekly_km_target", 30)),
        step=1.0,
    )

    if st.button("Save Profile"):
        save_profile({
            "gender": gender,
            "height_cm": height_cm,
            "start_weight_kg": start_weight,
            "target_weight_kg": target_weight,
            "weekly_km_target": weekly_km_target,
            "use_ideal_as_target": use_ideal_as_target,
        })
        st.success("Profile saved.")

    current_weight_for_calorie = latest_weight_or_start(df, start_weight)
    weekly_calorie_estimate = estimate_calories(weekly_km_target, current_weight_for_calorie, True)

    st.divider()
    st.caption("Calorie target is calculated automatically from your weekly distance target.")
    st.metric("Estimated weekly calorie target", f"{weekly_calorie_estimate:.0f} kcal")
    st.caption("Formula: body weight × running distance. Apple Health calories are used when available.")

df = fill_missing_calories(df, start_weight)

tabs = st.tabs([
    "🏠 Home",
    "➕ Input",
    "📊 Analytics",
    "📅 Calendar",
    "💾 Data"
])

# ============================================================
# Home
# ============================================================

with tabs[0]:
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
        st.markdown('<div class="section-title">Body Profile & Ideal Weight</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("Height", f"{height_cm:.0f} cm")
        p2.metric("BMI", f"{current_bmi:.1f}", bmi_category(current_bmi))
        p3.metric("Ideal weight", f"{ideal_weight:.1f} kg")
        st.caption(f"Healthy weight range based on BMI 18.5–24.9: {healthy_min:.1f}–{healthy_max:.1f} kg.")

        st.markdown('<div class="section-title">Weight-Loss Progress</div>', unsafe_allow_html=True)
        if pd.isna(latest_weight):
            st.info("No body-weight data yet. Add your weight in the Input tab.")
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
        st.info("No data yet. Start from the Input tab.")
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
        'Calories are estimated from body weight and running distance unless Apple Health or another source is added later.'
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

            entry = {
                "date": str(input_date),
                "ran": ran,
                "distance_km": distance_km if ran else 0,
                "duration_min": duration_min if ran else 0,
                "pace_min_per_km": pace if ran else None,
                "weight_kg": weight_kg if weight_kg > 0 else None,
                "calories_kcal": calories if ran else 0,
                "source": "Manual",
                "notes": notes
            }

            save_entry(entry)
            st.success(f"Saved ✅ Pace: {pace_to_text(pace)} | Estimated calories: {calories:.0f} kcal")
            st.rerun()
        except Exception as e:
            st.error(f"Could not save this entry: {e}")

# ============================================================
# Analytics
# ============================================================

with tabs[2]:
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data yet.")
    else:
        dfp = df.sort_values("date").copy()
        dfp["date"] = pd.to_datetime(dfp["date"])
        run_df = dfp[dfp["ran"] == True].copy()

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

        st.divider()

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
        st.caption("Calories are estimated automatically from weight and distance.")

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

# ============================================================
# Calendar
# ============================================================

with tabs[3]:
    st.markdown('<div class="section-title">Consistency Calendar</div>', unsafe_allow_html=True)

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
        year_options = list(range(today.year - 2, today.year + 2))
        selected_year = st.selectbox(
            "Year",
            options=year_options,
            index=year_options.index(today.year),
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

with tabs[4]:
    st.markdown('<div class="section-title">Saved Data</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data yet.")
    else:
        display_df = df.sort_values("date", ascending=False).copy()
        display_df["pace"] = display_df["pace_min_per_km"].apply(pace_to_text)
        display_df["duration"] = display_df["duration_min"].apply(duration_to_text)

        ordered_columns = [
            "id", "date", "ran", "distance_km", "duration", "pace",
            "weight_kg", "calories_kcal", "source", "notes"
        ]
        st.dataframe(display_df[ordered_columns], use_container_width=True)

        st.markdown('<div class="section-title">Delete Entry</div>', unsafe_allow_html=True)
        delete_id = st.text_input("Paste entry ID to delete")
        if st.button("Delete Selected Entry"):
            if delete_id:
                try:
                    delete_entry(delete_id)
                    st.success("Entry deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete entry: {e}")
            else:
                st.warning("Please paste an entry ID first.")
