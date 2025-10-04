# data_loaders.py
import os
import re
from pathlib import Path
from datetime import time, datetime

import pandas as pd

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


COURSE_ID_PATTERN = re.compile(r"[A-Z]{2,5}\s?-?\d{3,4}")


def _normalize_course_id(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    match = COURSE_ID_PATTERN.search(raw.upper())
    if not match:
        return None
    return re.sub(r"\s+", "-", match.group())


def load_degree_plan_credits(path: str) -> pd.DataFrame:
    """Extract course credits from the degree plan workbook."""
    file_path = Path(path)
    if not file_path.exists() or file_path.name.startswith("~$"):
        return pd.DataFrame(columns=["course_id", "credits"])

    records: list[tuple[str, float]] = []
    xls = pd.ExcelFile(file_path)

    for sheet in xls.sheet_names:
        raw = xls.parse(sheet, header=None)
        if raw.empty:
            continue

        header_idx = None
        for idx, row in raw.iterrows():
            values = row.astype(str).str.strip()
            if "Credits" in values.values:
                header_idx = idx
                break
        if header_idx is None:
            continue

        header = raw.iloc[header_idx].astype(str).str.strip()
        df = raw.iloc[header_idx + 1 :].reset_index(drop=True)
        df.columns = header

        credits_col = next((c for c in df.columns if str(c).strip().lower() == "credits"), None)
        if credits_col is None:
            continue

        name_col = None
        for candidate in df.columns:
            label = str(candidate).strip().lower()
            if label in {"course", "degree requirement"}:
                name_col = candidate
                break
        if name_col is None:
            continue

        course_names = df[name_col].astype(str).str.strip()
        credit_values = pd.to_numeric(df[credits_col], errors="coerce")

        for raw_name, credit in zip(course_names, credit_values):
            if pd.isna(credit):
                continue
            course_id = _normalize_course_id(raw_name)
            if not course_id:
                continue
            records.append((course_id, float(credit)))

    if not records:
        return pd.DataFrame(columns=["course_id", "credits"])

    credits_df = pd.DataFrame(records, columns=["course_id", "credits"])
    return credits_df.groupby("course_id", as_index=False)["credits"].max()

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

def load_course_catalog(
    path: str,
    sheet_name: str | int = 0,
    degree_plan_path: str | None = None,
) -> pd.DataFrame:
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

    if "credits" not in df.columns:
        df["credits"] = 0.0
    else:
        df["credits"] = pd.to_numeric(df["credits"], errors="coerce").fillna(0.0)

    plan_path = degree_plan_path or os.getenv(
        "DEGREE_PLAN_PATH", "sources/Degree Plan Templates/Degree Plan Original.xlsx"
    )
    if plan_path:
        plan_df = load_degree_plan_credits(plan_path)
        if not plan_df.empty:
            df = df.merge(plan_df, on="course_id", how="left", suffixes=("", "_plan"))
            df["credits"] = df["credits_plan"].fillna(df["credits"])
            df = df.drop(columns=[col for col in df.columns if col.endswith("_plan")])

    # No prerequisites column here; keep it blank so downstream logic works
    df["prerequisites"] = ""

    # Normalize instructor / modality strings
    df["instructor"] = df["instructor"].fillna("").astype(str).str.strip()
    df["modality"] = df["modality"].fillna("").astype(str).str.strip()

    # Select and order the fields your app expects, keeping extras if useful
    columns_for_app = [
        "course_id", "title", "credits", "days",
        "start_time", "end_time", "prerequisites",
        "instructor", "instructor_email", "term", "campus", "modality",
        "section", "class_number", "location", "room",
        "building", "component", "session", "topic",
        "subject", "catalog_number"
    ]
    return df.reindex(columns=columns_for_app, fill_value="")
