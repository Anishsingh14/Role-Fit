"""
visualize.py
------------
Part of Role-Fit — Smart Resume-to-Job Matching & Skill Gap Analysis.

Generates all charts for the project as PNG files (no frontend needed).
Every function saves an image into the given output directory and returns
the file path.

Design system: a single cohesive palette, typography, and set of layout
conventions (rounded bars, soft gridlines, consistent title/subtitle/
footer treatment) are shared across all six charts so they read as one
polished report rather than six independently-styled plots.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for terminal / server use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from sklearn.decomposition import PCA

from matcher_core import JobMatch, SkillGapReport
from job_data import MASTER_SKILLS

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------

# Color palette — one primary (brand), one success, one warning/gap, and a
# neutral scale. Chosen for good contrast at both screen and print size.
PRIMARY = "#4F46E5"       # indigo — brand / "your resume" / neutral data
PRIMARY_DARK = "#3730A3"
PRIMARY_LIGHT = "#C7D2FE"
SUCCESS = "#10B981"       # emerald — matched / have this skill
GAP = "#F97316"           # orange — missing / gap / to learn next
GAP_DARK = "#C2410C"
NEUTRAL_900 = "#111827"   # near-black text
NEUTRAL_600 = "#6B7280"   # muted text / subtitles
NEUTRAL_300 = "#E5E7EB"   # gridlines / track backgrounds
NEUTRAL_100 = "#F9FAFB"   # panel background
WHITE = "#FFFFFF"

FONT_FAMILY = "DejaVu Sans"  # bundled with matplotlib on every platform

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": FONT_FAMILY,
    "font.size": 10.5,
    "text.color": NEUTRAL_900,
    "axes.edgecolor": NEUTRAL_300,
    "axes.labelcolor": NEUTRAL_600,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlecolor": NEUTRAL_900,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": False,
    "xtick.color": NEUTRAL_600,
    "ytick.color": NEUTRAL_900,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
})

BRAND_FOOTER = "Role-Fit  ·  K-NN Resume-to-Job Matching"


def _add_header(fig, title: str, subtitle: str | None = None, align: str = "left",
                 header_height_in: float = 0.95):
    """Draw a title + optional subtitle directly in figure space (not axes
    space), so the header never overlaps the plot regardless of chart type
    (cartesian, polar, or pie). Reserves a fixed vertical band at the top
    of the figure for it.
    """
    fig_h = fig.get_size_inches()[1]
    top = 1 - header_height_in / fig_h
    fig.subplots_adjust(top=top)

    x = 0.5 if align == "center" else 0.02
    ha = "center" if align == "center" else "left"

    fig.text(x, 1 - 0.32 / fig_h, title, fontsize=17, fontweight="bold",
              color=NEUTRAL_900, ha=ha, va="top")
    if subtitle:
        fig.text(x, 1 - 0.66 / fig_h, subtitle, fontsize=10.5,
                  color=NEUTRAL_600, ha=ha, va="top")


def _add_footer(fig, extra: str | None = None):
    text = f"{BRAND_FOOTER}  ·  {extra}" if extra else BRAND_FOOTER
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
              fontsize=7.5, color=NEUTRAL_300 if not extra else NEUTRAL_600,
              style="italic")


def _rounded_hbar(ax, y_positions, widths, height=0.62, color=PRIMARY, zorder=3):
    """Draw horizontal bars with fully rounded ends (a flat rectangle bar
    reads as dated; a rounded 'pill' bar is the modern dashboard look)."""
    for y, w in zip(y_positions, widths):
        w = max(w, 1e-6)  # avoid a degenerate zero-width patch
        rounding = height / 2
        patch = FancyBboxPatch(
            (0, y - height / 2), w, height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0, facecolor=color, zorder=zorder,
        )
        ax.add_patch(patch)


def _save(fig, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CHART 1: TOP JOB MATCHES
# ---------------------------------------------------------------------------

def plot_top_matches_bar(matches: List[JobMatch], out_dir: str) -> str:
    """Bar chart of cosine similarity scores for top-K matched jobs."""
    titles = [m.title for m in matches][::-1]
    scores = [m.similarity * 100 for m in matches][::-1]
    n = len(titles)

    fig, ax = plt.subplots(figsize=(8, 0.85 * n + 1.3))
    y_pos = np.arange(n)

    # Faint full-width track behind each bar, like a progress bar, so the
    # eye immediately reads "percentage of a whole" rather than "raw length".
    _rounded_hbar(ax, y_pos, [100] * n, color=NEUTRAL_100, zorder=1)
    # Color the top match distinctly to draw the eye to the headline result
    colors = [PRIMARY_DARK if i == n - 1 else PRIMARY for i in range(n)]
    for y, w, c in zip(y_pos, scores, colors):
        _rounded_hbar(ax, [y], [w], color=c)

    for y, score in zip(y_pos, scores):
        ax.text(score + 2.5, y, f"{score:.1f}%", va="center", ha="left",
                 fontsize=10.5, fontweight="bold", color=NEUTRAL_900)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(titles, fontsize=11)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.7, n - 1 + 0.7)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.xaxis.grid(True, color=NEUTRAL_300, linestyle="-", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False)

    _add_header(fig, "Top Job Matches",
                f"Ranked by K-NN cosine similarity across {n} candidate roles")
    _add_footer(fig)

    return _save(fig, out_dir, "01_top_job_matches.png")


# ---------------------------------------------------------------------------
# CHART 2: SKILL MATCH VS GAP
# ---------------------------------------------------------------------------

def plot_skill_gap_bar(report: SkillGapReport, out_dir: str) -> str:
    """Horizontal 'chip' list: matched vs missing skills for the top job."""
    all_skills = report.matched_skills + report.missing_skills
    is_matched = [True] * len(report.matched_skills) + [False] * len(report.missing_skills)
    n = len(all_skills)

    fig, ax = plt.subplots(figsize=(7.5, 0.52 * n + 1.4))
    y_pos = np.arange(n)[::-1]  # top of list = first skill

    colors = [SUCCESS if m else GAP for m in is_matched]
    _rounded_hbar(ax, y_pos, [1] * n, height=0.68, color=NEUTRAL_100, zorder=1)
    for y, c in zip(y_pos, colors):
        _rounded_hbar(ax, [y], [1], height=0.68, color=c)

    for y, skill, matched in zip(y_pos, all_skills, is_matched):
        icon = "✓" if matched else "✕"
        ax.text(0.03, y, f"{icon}  {skill}", va="center", ha="left",
                 fontsize=10.5, fontweight="medium",
                 color=WHITE, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])

    _add_header(
        fig, "Skill Match vs. Gap",
        f"Target role: {report.job_title}   ·   {report.readiness_score:.0f}% ready   "
        f"·   {len(report.matched_skills)} matched, {len(report.missing_skills)} missing",
        header_height_in=1.05,
    )
    _add_footer(fig)

    return _save(fig, out_dir, "02_skill_gap_bar.png")


# ---------------------------------------------------------------------------
# CHART 3: READINESS DONUT
# ---------------------------------------------------------------------------

def _readiness_color(score: float) -> str:
    if score >= 75:
        return SUCCESS
    if score >= 45:
        return GAP
    return "#EF4444"  # red — low readiness


def plot_readiness_donut(report: SkillGapReport, out_dir: str) -> str:
    """Donut chart showing readiness % for the top matched job."""
    readiness = report.readiness_score
    remaining = 100 - readiness
    ring_color = _readiness_color(readiness)

    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.pie(
        [readiness, remaining],
        colors=[ring_color, NEUTRAL_300],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.32, edgecolor=WHITE, linewidth=3),
    )
    ax.text(0, 0.08, f"{readiness:.0f}%", ha="center", va="center",
             fontsize=34, fontweight="bold", color=NEUTRAL_900)
    ax.text(0, -0.16, "READY", ha="center", va="center",
             fontsize=11, color=NEUTRAL_600, fontweight="medium")

    ax.set_title("Readiness Score", loc="center", pad=20)
    _add_footer(fig, extra=f"Target role: {report.job_title}")

    return _save(fig, out_dir, "03_readiness_donut.png")


# ---------------------------------------------------------------------------
# CHART 4: SKILL COVERAGE RADAR
# ---------------------------------------------------------------------------

def plot_radar_chart(resume_skills: List[str], match: JobMatch, out_dir: str) -> str:
    """Radar chart comparing resume skill coverage vs job requirement for
    the top matched job's required skills."""
    skills = list(match.required_skills.keys())
    if len(skills) < 3:
        skills = (skills * 3)[:3]

    resume_set = set(resume_skills)
    max_weight = max(match.required_skills.values())

    job_values = [match.required_skills[s] / max_weight for s in skills]
    resume_values = [1.0 if s in resume_set else 0.0 for s in skills]

    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    job_values += job_values[:1]
    resume_values += resume_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(WHITE)

    ax.set_facecolor(NEUTRAL_100)
    ax.plot(angles, resume_values, color=PRIMARY, linewidth=2.4, label="Your Resume", zorder=3)
    ax.fill(angles, resume_values, color=PRIMARY, alpha=0.28, zorder=2)
    ax.plot(angles, job_values, color=GAP_DARK, linewidth=2, linestyle="--",
             label="Job Requirement", zorder=3)
    ax.fill(angles, job_values, color=GAP, alpha=0.12, zorder=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills, fontsize=9.5, color=NEUTRAL_900)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(NEUTRAL_300)
    ax.grid(color=NEUTRAL_300, linewidth=0.8)

    ax.set_title("Skill Coverage Radar", pad=36, fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), frameon=False, fontsize=10)
    fig.text(0.5, 0.02, f"{BRAND_FOOTER}  ·  Target: {match.title}", ha="center",
              fontsize=8, color=NEUTRAL_600, style="italic")

    return _save(fig, out_dir, "04_skill_radar.png")


# ---------------------------------------------------------------------------
# CHART 5: HIGHEST-IMPACT MISSING SKILLS
# ---------------------------------------------------------------------------

def plot_common_missing_skills(common_missing: List[Tuple[str, int]], out_dir: str) -> str:
    """Bar chart: skills most frequently missing across ALL top-K matched
    jobs — i.e. the highest-impact skills to learn next."""
    if not common_missing:
        fig, ax = plt.subplots(figsize=(7.5, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.55, "🎉  No skill gaps detected", ha="center", va="center",
                 fontsize=16, fontweight="bold", color=SUCCESS)
        ax.text(0.5, 0.25, "You already cover every required skill across your top matches.",
                 ha="center", va="center", fontsize=10.5, color=NEUTRAL_600)
        _add_header(fig, "Highest-Impact Skills to Learn Next", align="center", header_height_in=0.75)
        _add_footer(fig)
        return _save(fig, out_dir, "05_highest_impact_missing_skills.png")

    skills, counts = zip(*common_missing)
    n = len(skills)
    max_count = max(counts)

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.4))
    y_pos = np.arange(n)[::-1]

    _rounded_hbar(ax, y_pos, [max_count] * n, height=0.62, color=NEUTRAL_100, zorder=1)
    # Darker shade for higher-impact (more frequently missing) skills
    for y, c in zip(y_pos, counts):
        shade = GAP_DARK if c == max_count else GAP
        _rounded_hbar(ax, [y], [c], height=0.62, color=shade)

    for y, skill, c in zip(y_pos, skills, counts):
        ax.text(c + max_count * 0.03, y, str(c), va="center", ha="left",
                 fontsize=10.5, fontweight="bold", color=NEUTRAL_900)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(skills, fontsize=11)
    ax.set_xlim(0, max_count * 1.18)
    ax.set_ylim(-0.7, n - 1 + 0.7)
    ax.set_xticks([])
    ax.tick_params(left=False)

    _add_header(fig, "Highest-Impact Skills to Learn Next",
                "Number of top-matched roles that require each missing skill")
    _add_footer(fig)

    return _save(fig, out_dir, "05_highest_impact_missing_skills.png")


# ---------------------------------------------------------------------------
# CHART 6: 2D JOB-SPACE PROJECTION (PCA)
# ---------------------------------------------------------------------------

def plot_pca_projection(resume_vector: np.ndarray, matcher, out_dir: str, matches: List[JobMatch] | None = None) -> str:
    """2D PCA projection of all job vectors + the resume vector, to visually
    show which 'cluster' of jobs the resume lands closest to.

    Vectors are L2-normalized before PCA so the projection reflects skill
    *composition* (which skills, in what proportion) rather than raw skill
    *count* — a resume listing many skills would otherwise appear as a
    distant outlier purely due to vector magnitude, not job-fit direction.
    """
    all_vectors = np.vstack([matcher.job_matrix, resume_vector])
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_vectors_normalized = all_vectors / norms

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_vectors_normalized)

    job_coords = coords[:-1]
    resume_coord = coords[-1]

    matched_titles = {m.title for m in matches} if matches else set()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_facecolor(NEUTRAL_100)
    ax.grid(color=WHITE, linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, title in enumerate(matcher.job_titles):
        is_match = title in matched_titles
        color = PRIMARY if is_match else NEUTRAL_600
        alpha = 1.0 if is_match else 0.45
        size = 130 if is_match else 70
        ax.scatter(job_coords[i, 0], job_coords[i, 1], color=color, s=size,
                   alpha=alpha, edgecolors=WHITE, linewidth=1.1, zorder=3)
        ax.annotate(
            title, (job_coords[i, 0], job_coords[i, 1]),
            fontsize=8.5 if is_match else 7.5,
            fontweight="bold" if is_match else "normal",
            color=NEUTRAL_900 if is_match else NEUTRAL_600,
            alpha=1.0 if is_match else 0.7,
            xytext=(5, 4), textcoords="offset points", zorder=4,
        )

    ax.scatter(resume_coord[0], resume_coord[1], color=GAP, s=340,
               marker="*", label="Your Resume", edgecolors=WHITE, linewidth=1.2, zorder=5)

    legend_elems = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=GAP, markersize=16, label="Your Resume"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIMARY, markersize=10, label="Top-Matched Role"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NEUTRAL_600, markersize=8, alpha=0.6, label="Other Roles"),
    ]
    legend = ax.legend(handles=legend_elems, loc="upper left", frameon=True,
                        fontsize=9.5, facecolor=WHITE, edgecolor=NEUTRAL_300,
                        framealpha=0.95, borderpad=0.8)
    legend.get_frame().set_linewidth(0.8)

    ax.set_xlabel("Principal Component 1", fontsize=10)
    ax.set_ylabel("Principal Component 2", fontsize=10)
    ax.tick_params(labelsize=8, length=0)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.18)  # headroom for the legend box
    _add_header(fig, "Job-Space Map",
                "2D PCA projection — where your resume sits among all candidate roles",
                header_height_in=1.0)
    _add_footer(fig)

    return _save(fig, out_dir, "06_pca_job_space.png")


def generate_all_visuals(
    resume_skills: List[str],
    resume_vector: np.ndarray,
    matches: List[JobMatch],
    top_report: SkillGapReport,
    matcher,
    out_dir: str,
) -> List[str]:
    """Convenience function: generates every chart and returns list of paths."""
    from matcher_core import most_common_missing_skills
    common_missing = most_common_missing_skills(resume_skills, matches)

    paths = [
        plot_top_matches_bar(matches, out_dir),
        plot_skill_gap_bar(top_report, out_dir),
        plot_readiness_donut(top_report, out_dir),
        plot_radar_chart(resume_skills, matches[0], out_dir),
        plot_common_missing_skills(common_missing, out_dir),
        plot_pca_projection(resume_vector, matcher, out_dir, matches=matches),
    ]
    return paths
