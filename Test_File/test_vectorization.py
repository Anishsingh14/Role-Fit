"""
Unit tests for vectorize_resume() in matcher_core.py.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from matcher_core import vectorize_resume
from job_data import MASTER_SKILLS


class TestVectorizeResume:
    def test_vector_length_matches_master_skills(self):
        vector = vectorize_resume(["Python", "SQL"])
        assert len(vector) == len(MASTER_SKILLS)

    def test_known_skills_are_marked_one(self):
        vector = vectorize_resume(["Python", "SQL"])
        python_idx = MASTER_SKILLS.index("Python")
        sql_idx = MASTER_SKILLS.index("SQL")
        assert vector[python_idx] == 1.0
        assert vector[sql_idx] == 1.0

    def test_unmentioned_skills_are_zero(self):
        vector = vectorize_resume(["Python"])
        docker_idx = MASTER_SKILLS.index("Docker")
        assert vector[docker_idx] == 0.0

    def test_empty_skill_list_returns_all_zero_vector(self):
        vector = vectorize_resume([])
        assert np.sum(vector) == 0
        assert len(vector) == len(MASTER_SKILLS)

    def test_unknown_skill_name_is_silently_ignored(self):
        # A skill not in MASTER_SKILLS shouldn't raise or corrupt the vector
        vector = vectorize_resume(["Python", "SomeUnknownSkillXYZ"])
        assert np.sum(vector) == 1.0

    def test_duplicate_skills_do_not_double_count(self):
        vector = vectorize_resume(["Python", "Python", "Python"])
        python_idx = MASTER_SKILLS.index("Python")
        assert vector[python_idx] == 1.0
