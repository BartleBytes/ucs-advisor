# data_loaders.py
import pandas as pd
from datetime import time, datetime

COLUMN_MAP = {
    "Term": "term",
    "Subject": "subject",
    "Catalog Number": "catalog_number",
    "Class Section": "section",
    "Class Number": "class_number",
    "Course Title": "title",
    "Campus": "campus",
    "Academic Level": "level",
    "Session": "session",
    "Component": "component",
    "Topic": "topic",
    "Combined Section": "combined_section",
    "Combined Sections": "combined_sections",
    "Consent": "consent",
    "Start": "start_date",
    "End": "end_date",
    "Time Start": "time_start",
    "Time End": "time_end",
    "Pattern": "pattern",
    "Location": "location",
    "Building": "building",
    "Room": "room",
    "Instructor": "instructor",
    "Modality": "modality",
    "Instructor Email": "instructor_email",
    "Units": "credits",
}

def _normalize_time(value: object) -> str:
    """Return times as HH:MM (24h)."""
    if pd.isna(value):
        return ""
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    s = str(value).strip()
    if not s or s in {"0", "00:00", "00:00:00"}:
        return ""
    try:
        return pd.to_datetime(s).strftime("%H:%M")
    except Exception:
        return s  # fall back to the raw string

def load_course_catalog(path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    # Read the sheet and rename columns to our normalized names
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Identify the header row (expect first cell to be "Term")
    header_row = raw.index[raw.iloc[:, 0] == "Term"]
    if header_row.empty:
        raise ValueError("Could not locate header row with 'Term'.")

    header_idx = header_row[0]
    header = raw.iloc[header_idx]
    df = raw.iloc[header_idx + 1 :].reset_index(drop=True)
    df.columns = header

    df = df.rename(columns=COLUMN_MAP)

    # Build composite identifiers
    subject_raw = df["subject"].fillna("").astype(str).str.strip()
    df["subject"] = subject_raw.str.split("-").str[0].str.strip()
    df["catalog_number"] = df["catalog_number"].fillna("").astype(str).str.strip()
    df["course_id"] = df["subject"] + "-" + df["catalog_number"]


    # Days pattern becomes our "days" column (strip spaces)
    df["days"] = df["pattern"].fillna("").astype(str).str.replace(" ", "")

    # Start/end times in HH:MM
    df["start_time"] = df["time_start"].apply(_normalize_time)
    df["end_time"] = df["time_end"].apply(_normalize_time)

    # Credits are not present in this sheet—set to 0 (or pull from another file if available)
    if "credits" not in df.columns:
        df["credits"] = 0.0
    else:
        df["credits"] = pd.to_numeric(df["credits"], errors="coerce").fillna(0.0)

    # No prerequisites column here; keep it blank so downstream logic works
    df["prerequisites"] = ""

    # Normalize instructor / modality strings
    df["instructor"] = df["instructor"].fillna("").astype(str).str.strip()
    df["modality"] = df["modality"].fillna("").astype(str).str.strip()

    # Select and order the fields your app expects, keeping extras if useful
    columns_for_app = [
        "course_id", "title", "credits", "days",
        "start_time", "end_time", "prerequisites",
        "instructor", "term", "campus", "modality",
        "section", "class_number", "location", "room",
        "building", "component", "session", "topic",
        "subject", "catalog_number"
    ]
    return df.reindex(columns=columns_for_app, fill_value="")
