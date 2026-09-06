#Based_on_K-NN

import argparse
import json
import os
import sys
import textwrap
from dataclasses import asdict

from matcher_core import (
    parse_resume_file,
    extract_skills,
    vectorize_resume,
    ResumeJobMatcher,
    skill_gap_report,
    most_common_missing_skills,
)
from visualize import generate_all_visuals

def print_header(text: str):
    bar = "=" * 70
    print(f"\n{bar}\n{text}\n{bar}")

def print_section(text: str):
    print(f"\n--- {text} " + "-" * max(0, 55 - len(text)))

def prompt_for_resume_path() -> str:
    """Interactively ask the user for a resume file path, re-prompting on
    empty input, missing files, or unsupported extensions until a valid
    path is given."""
    while True:
        path = input("Provide the path of your Resume File ~ .pdf or .docx : ").strip()

        # Allow users to paste quoted paths (common on Windows/macOS)
        if len(path) >= 2 and path[0] == path[-1] and path[0] in ("'", '"'):
            path = path[1:-1]

        if not path:
            print("  Please enter a file path : \n")
            continue
        if not os.path.exists(path):
            print(f"  [!] No file found at: {path}\n      Please check the path and try again.\n")
            continue
        if os.path.splitext(path)[1].lower() not in (".pdf", ".docx"):
            print("  [!] Only .pdf and .docx files are supported. Please try again.\n")
            continue
        return path

def main():
    parser = argparse.ArgumentParser(
        description="Smart Resume-to-Job Matching & Skill Gap Analysis (K-NN based)"
    )
    parser.add_argument(
        "resume_path",
        nargs="?",
        default=None,
        help="Path to the resume file (.pdf or .docx). "
             "If omitted, you'll be prompted for it interactively.",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of top job matches to return (default: 5)"
    )
    parser.add_argument(
        "--out", type=str, default="output", help="Output directory for charts + JSON report"
    )
    args = parser.parse_args()

    print_header("ROLE-FIT — SMART RESUME-TO-JOB MATCHING & SKILL GAP ANALYSIS")

    # If no path was given on the command line, ask for it interactively.
    if args.resume_path:
        resume_path = args.resume_path
    else:
        try:
            resume_path = prompt_for_resume_path()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled. No resume path provided. Exiting.")
            sys.exit(1)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nInput resume : {resume_path}")
    print(f"Top-K matches: {args.k}")
    print(f"Output folder: {out_dir}/")

    # Stage 1: Parsing the resume file
    try:
        print_section("STAGE 1: Parsing resume file")
        raw_text = parse_resume_file(resume_path)
        print(f"Extracted {len(raw_text)} characters of text.")
    except Exception as e:
        print(f"\n[ERROR] Failed to parse resume: {e}")
        sys.exit(1)

    # Stage 2: Extracting skills
    print_section("STAGE 2: Extracting skills")
    resume_skills = extract_skills(raw_text)
    if not resume_skills:
        print("[WARNING] No known skills were detected in this resume. "
              "Results may be inaccurate. Check job_data.py's SKILL_SYNONYMS "
              "list to extend coverage.")
    else:
        print(f"Found {len(resume_skills)} skills:")
        print(textwrap.fill(", ".join(resume_skills), width=70))

    # Stage 3 & 4: Vectorization
    print_section("STAGE 3: Vectorizing resume & job dataset")
    resume_vector = vectorize_resume(resume_skills)
    matcher = ResumeJobMatcher()
    print(f"Job dataset loaded: {len(matcher.job_titles)} roles "
          f"across {matcher.job_matrix.shape[1]} tracked skills.")

    # Stage 5: K-NN Matching 
    print_section("STAGE 4: Running K-NN job matching")
    matches = matcher.find_top_matches(resume_vector, k=args.k)
    print(f"Top {len(matches)} matched roles:")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m.title:<32} similarity = {m.similarity * 100:5.1f}%")

    # Stage 6: Skill Gap Analysis (for top match) 
    print_section("STAGE 5: Skill gap analysis (top match)")
    top_report = skill_gap_report(resume_skills, matches[0])
    print(f"Target role     : {top_report.job_title}")
    print(f"Readiness score : {top_report.readiness_score}%")
    print(f"Matched skills  : {', '.join(top_report.matched_skills) or 'None'}")
    print(f"Missing skills  : {', '.join(top_report.missing_skills) or 'None'}")

    common_missing = most_common_missing_skills(resume_skills, matches)
    print_section("Highest-impact skills to learn (across top matches)")
    if common_missing:
        for skill, count in common_missing:
            print(f"  - {skill:<25} needed by {count}/{len(matches)} matched roles")
    else:
        print("  No gaps detected across top matches — great fit!")

    # Stage 7: Visualizations
    print_section("STAGE 6: Generating visualizations")
    chart_paths = generate_all_visuals(
        resume_skills, resume_vector, matches, top_report, matcher, out_dir
    )
    for p in chart_paths:
        print(f"  saved -> {p}")

    # Save your report (JSON) 
    report = {
        "resume_file": resume_path,
        "extracted_skills": resume_skills,
        "top_matches": [asdict(m) for m in matches],
        "top_match_skill_gap": asdict(top_report),
        "highest_impact_missing_skills": [
            {"skill": s, "needed_by_n_matches": c} for s, c in common_missing
        ],
        "charts_generated": chart_paths,
    }
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull JSON report saved -> {json_path}")

    print_header("DONE")

if __name__ == "__main__":
    main()
