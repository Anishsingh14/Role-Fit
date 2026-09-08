"""
Unit tests for extract_skills() in matcher_core.py.

Covers: exact synonym matching, case-insensitivity, word-boundary edge
cases (avoiding partial-word false positives), multi-word skill phrases,
and the fuzzy typo-tolerance pass.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matcher_core import extract_skills


class TestExactSynonymMatching:
    def test_finds_canonical_skill_directly(self):
        text = "Experienced with Python and SQL."
        skills = extract_skills(text, fuzzy=False)
        assert "Python" in skills
        assert "SQL" in skills

    def test_case_insensitive_matching(self):
        text = "PYTHON, python, PyThOn"
        skills = extract_skills(text, fuzzy=False)
        assert "Python" in skills

    def test_synonym_maps_to_canonical_name(self):
        # "js" is a synonym for the canonical "JavaScript"
        text = "5 years of experience with js and html"
        skills = extract_skills(text, fuzzy=False)
        assert "JavaScript" in skills
        assert "HTML/CSS" in skills

    def test_multi_word_skill_phrase(self):
        text = "Strong background in machine learning and data visualization."
        skills = extract_skills(text, fuzzy=False)
        assert "Machine Learning" in skills
        assert "Data Visualization" in skills

    def test_no_skills_found_returns_empty_list(self):
        text = "I enjoy hiking, painting, and playing the guitar."
        skills = extract_skills(text, fuzzy=False)
        assert skills == []


class TestWordBoundaryEdgeCases:
    def test_java_does_not_match_inside_javascript(self):
        # "java" should NOT be falsely detected inside "javascript"
        text = "I have experience with JavaScript development."
        skills = extract_skills(text, fuzzy=False)
        assert "JavaScript" in skills
        assert "Java" not in skills

    def test_r_programming_does_not_match_inside_other_words(self):
        # Guards against "r" matching inside words like "developer" or "server"
        text = "Worked as a backend developer managing servers."
        skills = extract_skills(text, fuzzy=False)
        assert "R Programming" not in skills

    def test_git_does_not_match_inside_digital(self):
        text = "Focused on digital marketing strategy."
        skills = extract_skills(text, fuzzy=False)
        assert "Git" not in skills


class TestFuzzyTypoTolerance:
    def test_catches_common_typo_with_fuzzy_enabled(self):
        # "Pyhton" is a one-letter-transposition typo of "python"
        text = "5 years of experience with Pyhton development."
        skills = extract_skills(text, fuzzy=True)
        assert "Python" in skills

    def test_fuzzy_disabled_does_not_catch_typo(self):
        text = "5 years of experience with Pyhton development."
        skills = extract_skills(text, fuzzy=False)
        assert "Python" not in skills

    def test_fuzzy_does_not_introduce_wild_false_positives(self):
        # Generic unrelated text should not spuriously match many skills
        text = "The quick brown fox jumps over the lazy dog near the river."
        skills = extract_skills(text, fuzzy=True)
        assert len(skills) == 0

    def test_single_word_does_not_falsely_match_multiword_synonym(self):
        # Regression test: "programming" was incorrectly fuzzy-matching
        # the multi-word synonym "r programming" because it is a near-total
        # substring of it, inflating difflib's similarity ratio. Word-count
        # bucketing must prevent this.
        text = "Strong background in software programming and coding."
        skills = extract_skills(text, fuzzy=True)
        assert "R Programming" not in skills
