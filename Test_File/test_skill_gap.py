"""
Unit tests for skill_gap_report() and most_common_missing_skills()
in matcher_core.py.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import pytest

from matcher_core import JobMatch, skill_gap_report, most_common_missing_skills


def make_job_match(required_skills):
    return JobMatch(title="Test Role", similarity=0.75, required_skills=required_skills)


class TestSkillGapReport:
    def test_full_match_gives_100_percent_readiness(self):
        job = make_job_match({"Python": 3, "SQL": 2})
        report = skill_gap_report(["Python", "SQL"], job)
        assert report.readiness_score == 100.0
        assert report.missing_skills == []
        assert set(report.matched_skills) == {"Python", "SQL"}

    def test_no_match_gives_0_percent_readiness(self):
        job = make_job_match({"Python": 3, "SQL": 2})
        report = skill_gap_report(["Docker", "Kubernetes"], job)
        assert report.readiness_score == 0.0
        assert set(report.missing_skills) == {"Python", "SQL"}
        assert report.matched_skills == []

    def test_partial_match_weighted_correctly(self):
        # Python (weight 3) matched, SQL (weight 2) missing
        # readiness = 3 / (3+2) * 100 = 60.0
        job = make_job_match({"Python": 3, "SQL": 2})
        report = skill_gap_report(["Python"], job)
        assert report.readiness_score == 60.0

    def test_missing_skills_sorted_by_importance_descending(self):
        job = make_job_match({"Python": 1, "SQL": 3, "Docker": 2})
        report = skill_gap_report([], job)
        # Highest weight (SQL=3) should come first
        assert report.missing_skills[0] == "SQL"
        assert report.missing_skills[-1] == "Python"

    def test_matched_skills_sorted_by_importance_descending(self):
        job = make_job_match({"Python": 1, "SQL": 3, "Docker": 2})
        report = skill_gap_report(["Python", "SQL", "Docker"], job)
        assert report.matched_skills[0] == "SQL"
        assert report.matched_skills[-1] == "Python"

    def test_empty_required_skills_warns_and_returns_zero(self):
        # A job with no required skills is a data problem — should warn
        # explicitly rather than silently defaulting.
        job = make_job_match({})
        with pytest.warns(UserWarning, match="no required skills"):
            report = skill_gap_report(["Python"], job)
        assert report.readiness_score == 0.0
        assert report.matched_skills == []
        assert report.missing_skills == []


class TestMostCommonMissingSkills:
    def test_counts_missing_skills_across_multiple_jobs(self):
        matches = [
            make_job_match({"Docker": 2, "Python": 3}),
            make_job_match({"Docker": 2, "SQL": 1}),
            make_job_match({"Docker": 3}),
        ]
        result = most_common_missing_skills([], matches)
        result_dict = dict(result)
        # Docker is missing from all 3 jobs -> should be the top result
        assert result[0][0] == "Docker"
        assert result_dict["Docker"] == 3

    def test_resume_skills_are_excluded_from_missing(self):
        matches = [make_job_match({"Python": 3, "Docker": 2})]
        result = most_common_missing_skills(["Python"], matches)
        result_skills = [s for s, _ in result]
        assert "Python" not in result_skills
        assert "Docker" in result_skills

    def test_no_gaps_returns_empty_list(self):
        matches = [make_job_match({"Python": 3})]
        result = most_common_missing_skills(["Python"], matches)
        assert result == []
