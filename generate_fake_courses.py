# generate_fake_courses.py
import csv, random, itertools, os
random.seed(42)

OUT_PATH = "data/courses.csv"

DEPARTMENTS = {
    "CSCI": {
        "titles": {
            1000: ["Fundamentals of Computing", "Intro to CS", "Computational Thinking"],
            2000: ["Object-Oriented Programming", "Data Structures", "Computer Systems"],
            3000: ["Algorithms", "Database Systems", "Operating Systems", "Networks"],
            4000: ["Software Engineering", "Machine Learning", "Distributed Systems"],
            5000: ["Advanced Machine Learning", "NLP", "Cloud & DevOps"],
        }
    },
    "MATH": {
        "titles": {
            1000: ["College Algebra", "Precalculus"],
            1400: ["Calculus I"],         # Nonstandard but handy for prereq chain
            2400: ["Calculus II"],
            3000: ["Linear Algebra", "Discrete Mathematics", "Probability"],
            4000: ["Numerical Analysis", "Real Analysis"],
            5000: ["Optimization", "Stochastic Processes"],
        }
    },
    "BANA": {
        "titles": {
            6000: ["Statistics for Business Analytics", "Data Management for Business"],
            6610: ["Advanced Analytics", "Experiment Design", "Forecasting & Time Series"],
            6620: ["Data Visualization", "Big Data Systems"],
        }
    },
    "ECON": {
        "titles": {
            1000: ["Principles of Microeconomics", "Principles of Macroeconomics"],
            3000: ["Intermediate Microeconomics", "Intermediate Macroeconomics", "Econometrics"],
            4000: ["Game Theory", "Public Finance"],
            5000: ["Advanced Econometrics", "Policy Evaluation"],
        }
    },
    "ISMG": {
        "titles": {
            3000: ["Business Programming", "Systems Analysis & Design"],
            4000: ["Project Management", "Enterprise Data Warehousing"],
            6000: ["IT Strategy", "Analytics in Enterprise"],
        }
    }
}

INSTRUCTORS = [
    "Dr. Smith","Dr. Lee","Dr. Chen","Dr. Gomez","Dr. Nguyen","Prof. Patel","Prof. Romero",
    "Dr. Hart","Dr. Zhou","Dr. Park","Dr. Kim","Dr. Shah","Dr. Allen","Dr. Rivera","Dr. Khan"
]

TERMS = ["Fall 2025", "Spring 2026"]
CAMPUSES = ["Denver", "Auraria Online"]
MODALITIES = ["In-Person", "Hybrid", "Online"]
DAYS_PATTERNS = ["MWF", "TR", "MW", "WF", "T", "R"]  # keep simple; you can expand later

# 50-minute & 75-minute blocks; hour ranges to keep schedules realistic
SLOTS_50 = ["08:00-08:50","09:00-09:50","10:00-10:50","11:00-11:50","12:00-12:50","13:00-13:50","14:00-14:50"]
SLOTS_75 = ["08:30-09:45","10:00-11:15","11:30-12:45","13:00-14:15","14:30-15:45","16:00-17:15"]

def choose_time(days):
    if days == "MWF":
        start, end = random.choice(SLOTS_50).split("-")
        return start, end
    else:  # TR, MW, WF, T, R -> 75 min by default
        start, end = random.choice(SLOTS_75).split("-")
        return start, end

def level_chain(levels_sorted):
    """Create chained sequences for prereqs within a department."""
    chains = []
    prev = None
    for lvl in levels_sorted:
        chains.append((prev, lvl))
        prev = lvl
    return chains  # e.g., (None,1000)->(1000,2000)->(2000,3000)...

def make_course_id(prefix, level, seq):
    # e.g., CSCI-2312 / MATH-1401 / BANA-6610
    return f"{prefix}-{level + seq:04d}" if level < 1000 or level % 1000 != 0 else f"{prefix}-{level}"

def generate_department_courses():
    rows = []
    for dept, meta in DEPARTMENTS.items():
        level_keys = sorted(meta["titles"].keys())
        chains = level_chain(level_keys)  # (prev_level, current_level)
        seq_counter = 0

        for prev_lvl, lvl in chains:
            titles = meta["titles"][lvl]
            # Make ~8 courses at each level (tweak for variety)
            for _ in range(8):
                title = random.choice(titles)
                days = random.choice(DAYS_PATTERNS)
                start, end = choose_time(days)
                credits = random.choice([3, 3, 3, 4]) if dept != "BANA" else 3
                instructor = random.choice(INSTRUCTORS)
                term = random.choice(TERMS)
                campus = random.choice(CAMPUSES)
                modality = random.choice(MODALITIES)

                # Prereqs: if there is a prev level, pick 0–2 prereqs from that level’s titles
                prereqs = ""
                if prev_lvl is not None:
                    prev_titles = meta["titles"][prev_lvl]
                    # fabricate some prev course IDs that could exist
                    possible_prev_ids = []
                    for i in range(1, 7):
                        cid = make_course_id(dept, prev_lvl, i)
                        possible_prev_ids.append(cid)
                    k = random.choice([0, 1, 1, 2])  # more likely to have 0–1 prereqs
                    prereqs = ",".join(random.sample(possible_prev_ids, k)) if k else ""

                seq_counter = (seq_counter + 1) % 50  # just to vary IDs a bit
                course_id = make_course_id(dept, lvl, seq_counter)

                rows.append({
                    "course_id": course_id,
                    "title": title,
                    "credits": credits,
                    "days": days,
                    "start_time": start,
                    "end_time": end,
                    "instructor": instructor,
                    "prerequisites": prereqs,
                    "term": term,
                    "campus": campus,
                    "modality": modality
                })
    return rows

def main():
    os.makedirs("data", exist_ok=True)
    rows = generate_department_courses()
    # Deduplicate on course_id if collisions (unlikely but safe)
    seen = set()
    deduped = []
    for r in rows:
        if r["course_id"] in seen:
            continue
        seen.add(r["course_id"])
        deduped.append(r)

    # Sort for nicer CSV
    deduped.sort(key=lambda r: (r["course_id"], r["term"]))
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "course_id","title","credits","days","start_time","end_time",
            "instructor","prerequisites","term","campus","modality"
        ])
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Wrote {len(deduped)} courses to {OUT_PATH}")

if __name__ == "__main__":
    main()
