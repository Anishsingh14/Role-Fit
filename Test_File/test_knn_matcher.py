"""
Unit tests for ResumeJobMatcher.find_top_matches() in matcher_core.py.

Uses a small, hand-crafted mock job dataset (via monkeypatch) instead of
the full real dataset, so the test is fast, deterministic, and independent
of any future changes to job_data.py's real job list.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matcher_core
from matcher_core import ResumeJobMatcher, vectorize_resume
from job_data import MASTER_SKILLS


MOCK_JOBS = [
    {
        "title": "Mock Python Role",
        "required_skills": {"Python": 3, "SQL": 2},
    },
    {
        "title": "Mock Frontend Role",
        "required_skills": {"JavaScript": 3, "HTML/CSS": 3, "React": 2},
    },
    {
        "title": "Mock DevOps Role",
        "required_skills": {"Docker": 3, "Kubernetes": 3, "Linux": 2},
    },
]


def mock_get_job_dataset():
    return MOCK_JOBS


class TestResumeJobMatcher:
    def test_matcher_loads_mock_dataset(self, monkeypatch):
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()
        assert matcher.job_titles == [
            "Mock Python Role", "Mock Frontend Role", "Mock DevOps Role"
        ]
        assert matcher.job_matrix.shape == (3, len(MASTER_SKILLS))

    def test_python_resume_matches_python_role_first(self, monkeypatch):
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()

        resume_vector = vectorize_resume(["Python", "SQL"])
        matches = matcher.find_top_matches(resume_vector, k=3)

        assert len(matches) == 3
        assert matches[0].title == "Mock Python Role"
        # The resume has exactly the two skills the top job needs (Python,
        # SQL), so it should score highest of the three — but not
        # necessarily a perfect 1.0, since the job vector is weighted
        # (Python=3, SQL=2) while the resume vector is binary (1, 1), and
        # cosine similarity between a weighted and unweighted vector over
        # the same support is high but not identical in direction.
        assert matches[0].similarity > 0.95
        assert matches[0].similarity > matches[1].similarity
        assert matches[0].similarity > matches[2].similarity

    def test_frontend_resume_matches_frontend_role_first(self, monkeypatch):
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()

        resume_vector = vectorize_resume(["JavaScript", "HTML/CSS", "React"])
        matches = matcher.find_top_matches(resume_vector, k=1)

        assert matches[0].title == "Mock Frontend Role"

    def test_k_is_capped_at_available_job_count(self, monkeypatch):
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()

        resume_vector = vectorize_resume(["Python"])
        # Ask for more matches than exist in the mock dataset (3 jobs)
        matches = matcher.find_top_matches(resume_vector, k=10)
        assert len(matches) == 3

    def test_empty_resume_still_returns_matches_without_crashing(self, monkeypatch):
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()

        resume_vector = vectorize_resume([])
        matches = matcher.find_top_matches(resume_vector, k=3)
        assert len(matches) == 3

    def test_weighted_matrix_used_not_binarized(self, monkeypatch):
        # Regression test for Priority Fix #1: the job matrix fitted into
        # NearestNeighbors must retain importance weights (not be flattened
        # to 1s and 0s), so "must-have" skills count more than
        # "nice-to-have" ones during matching.
        monkeypatch.setattr(matcher_core, "get_job_dataset", mock_get_job_dataset)
        matcher = ResumeJobMatcher()

        python_idx = MASTER_SKILLS.index("Python")
        sql_idx = MASTER_SKILLS.index("SQL")
        # Mock Python Role has Python=3, SQL=2 -> matrix should preserve those
        assert matcher.job_matrix[0, python_idx] == 3
        assert matcher.job_matrix[0, sql_idx] == 2
        # Confirm it's genuinely weighted, not binarized to 1/0
        assert not np.array_equal(
            matcher.job_matrix, (matcher.job_matrix > 0).astype(float)
        )
