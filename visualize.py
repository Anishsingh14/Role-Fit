"""
Generates all charts for the project as PNG files.

"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for terminal / server use
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from matcher_core import JobMatch, SkillGapReport
from job_data import MASTER_SKILLS

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

ACCENT = "#4C6FFF"
ACCENT_DARK = "#2A3F9D"
GAP_COLOR = "#E4572E"
MATCH_COLOR = "#2FA84F"


def _save(fig, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def plot_top_matches_bar(matches: List[JobMatch], out_dir: str) -> str:
    # Bar chart of cosine similarity scores for top-K matched jobs
    titles = [m.title for m in matches][::-1]
    scores = [m.similarity * 100 for m in matches][::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(titles, scores, color=ACCENT)
    ax.set_xlabel("Match Score (%)")
    ax.set_title("Top Job Matches (K-NN Cosine Similarity)", fontweight="bold")
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}%", va="center", fontsize=9)
    return _save(fig, out_dir, "01_top_job_matches.png")


def plot_skill_gap_bar(report: SkillGapReport, out_dir: str) -> str:
    # Horizontal bar chart: matched vs missing skills for the top job
    all_skills = report.matched_skills + report.missing_skills
    colors = [MATCH_COLOR] * len(report.matched_skills) + \
             [GAP_COLOR] * len(report.missing_skills)
    values = [1] * len(all_skills)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(all_skills))))
    ax.barh(all_skills[::-1], values[::-1], color=colors[::-1])
    ax.set_xticks([])
    ax.set_title(f"Skill Match vs Gap — {report.job_title}", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=MATCH_COLOR, label="You have this skill"),
        Patch(facecolor=GAP_COLOR, label="Missing skill"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")
    return _save(fig, out_dir, "02_skill_gap_bar.png")


def plot_readiness_donut(report: SkillGapReport, out_dir: str) -> str:
    # Donut chart showing readiness % for the top matched job
    readiness = report.readiness_score
    remaining = 100 - readiness

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, _ = ax.pie(
        [readiness, remaining],
        colors=[ACCENT, "#E5E7EB"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35),
    )
    ax.text(0, 0.05, f"{readiness:.0f}%", ha="center", va="center",
             fontsize=26, fontweight="bold", color=ACCENT_DARK)
    ax.text(0, -0.18, "Ready", ha="center", va="center",
             fontsize=11, color="#555555")
    ax.set_title(f"Readiness Score — {report.job_title}", fontweight="bold")
    return _save(fig, out_dir, "03_readiness_donut.png")


def plot_radar_chart(resume_skills: List[str], match: JobMatch, out_dir: str) -> str:
    """Radar chart comparing resume skill coverage vs job requirement for
    the top matched job's required skills."""
    skills = list(match.required_skills.keys())
    if len(skills) < 3:
        # Radar needs at least 3 axes to look meaningful; pad if necessary
        skills = (skills * 3)[:3]

    resume_set = set(resume_skills)
    max_weight = max(match.required_skills.values())

    job_values = [match.required_skills[s] / max_weight for s in skills]
    resume_values = [1.0 if s in resume_set else 0.0 for s in skills]

    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    job_values += job_values[:1]
    resume_values += resume_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, job_values, color=GAP_COLOR, linewidth=2, label="Job Requirement")
    ax.fill(angles, job_values, color=GAP_COLOR, alpha=0.15)
    ax.plot(angles, resume_values, color=ACCENT, linewidth=2, label="Your Resume")
    ax.fill(angles, resume_values, color=ACCENT, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"Skill Coverage Radar — {match.title}", fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    return _save(fig, out_dir, "04_skill_radar.png")


def plot_common_missing_skills(common_missing: List[Tuple[str, int]], out_dir: str) -> str:
    """Bar chart: skills most frequently missing across ALL top-K matched
    jobs i.e. the highest-impact skills to learn next."""
    if not common_missing:
        skills, counts = ["No gaps found!"], [0]
    else:
        skills, counts = zip(*common_missing)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(skills))))
    bars = ax.barh(list(skills)[::-1], list(counts)[::-1], color=GAP_COLOR)
    ax.set_xlabel("Appears in how many of your top matched jobs")
    ax.set_title("Highest-Impact Skills To Learn Next", fontweight="bold")
    for bar, c in zip(bars, list(counts)[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 str(c), va="center", fontsize=9)
    return _save(fig, out_dir, "05_highest_impact_missing_skills.png")


def plot_pca_projection(resume_vector: np.ndarray, matcher, out_dir: str) -> str:
    """2D PCA projection of all job vectors + the resume vector, to visually
    show which 'cluster' of jobs the resume lands closest to.
    Vectors are L2-normalized before PCA so the projection reflects skill
    *composition* (which skills, in what proportion) rather than raw skill
    *count* — a resume listing many skills would otherwise appear as a
    distant outlier purely due to vector magnitude, not job-fit direction.
    """
    all_vectors = np.vstack([matcher.job_matrix_binary, resume_vector])
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid divide-by-zero for an empty skill vector
    all_vectors_normalized = all_vectors / norms

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_vectors_normalized)

    job_coords = coords[:-1]
    resume_coord = coords[-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(job_coords[:, 0], job_coords[:, 1], color=ACCENT, s=90,
               label="Job Roles", edgecolors="white", linewidth=0.6)
    for i, title in enumerate(matcher.job_titles):
        ax.annotate(title, (job_coords[i, 0], job_coords[i, 1]),
                    fontsize=7, alpha=0.75, xytext=(4, 3), textcoords="offset points")

    ax.scatter(resume_coord[0], resume_coord[1], color=GAP_COLOR, s=220,
               marker="*", label="Your Resume", edgecolors="black", linewidth=0.6, zorder=5)
    ax.set_title("2D Job-Space Projection (PCA) — Where Your Resume Lands", fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="best")
    return _save(fig, out_dir, "06_pca_job_space.png")


def generate_all_visuals(
    resume_skills: List[str],
    resume_vector: np.ndarray,
    matches: List[JobMatch],
    top_report: SkillGapReport,
    matcher,
    out_dir: str,
) -> List[str]:
    # Convenience function: generates every chart and returns list of paths
    common_missing = None
    from matcher_core import most_common_missing_skills
    common_missing = most_common_missing_skills(resume_skills, matches)

    paths = [
        plot_top_matches_bar(matches, out_dir),
        plot_skill_gap_bar(top_report, out_dir),
        plot_readiness_donut(top_report, out_dir),
        plot_radar_chart(resume_skills, matches[0], out_dir),
        plot_common_missing_skills(common_missing, out_dir),
        plot_pca_projection(resume_vector, matcher, out_dir),
    ]
    return paths
