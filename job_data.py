"""
The "Database" for the project.

Instead of downloading an external dataset, we programmatically generate a
clean, labeled dataset of job roles with their required skills and skill
importance weights. This keeps the project fully self-contained and
reproducible (no internet download, no missing-file errors) while still
behaving like a real dataset that a K-NN model can be trained/queried against.

If you later want to swap this for a real-world dataset (e.g. a Kaggle
"LinkedIn Job Postings" CSV), you only need to replace `get_job_dataset()`
with a loader that returns the same structure.
"""

from __future__ import annotations
from typing import Dict, List

# 1. MASTER SKILLS TAXONOMY
"""
A curated list of tech + soft skills, each mapped to its accepted synonyms.
The extractor searches resume text for the canonical name OR any synonym,
and always records the canonical name. This solves the "Python" vs
"python3" / "JS" vs "JavaScript" / "ML" vs "Machine Learning" problem."""

SKILL_SYNONYMS: Dict[str, List[str]] = {
    "Python": ["python", "python3"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "t-sql", "pl/sql"],
    "R Programming": ["r programming", " r ", "r language"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "HTML/CSS": ["html", "css", "html5", "css3"],
    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Angular": ["angular", "angularjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "REST API": ["rest api", "restful", "rest apis", "api development"],
    "MongoDB": ["mongodb", "mongo db"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Linux": ["linux", "unix", "bash", "shell scripting"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment", "jenkins"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl", "neural networks"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv", "opencv"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data viz", "matplotlib", "seaborn", "tableau", "power bi"],
    "Statistics": ["statistics", "statistical analysis"],
    "Excel": ["excel", "ms excel", "spreadsheets"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop"],
    "Big Data": ["big data"],
    "Data Structures": ["data structures", "dsa", "algorithms"],
    "System Design": ["system design", "distributed systems"],
    "Microservices": ["microservices"],
    "Agile/Scrum": ["agile", "scrum", "kanban"],
    "Project Management": ["project management", "pmp"],
    "Communication": ["communication skills", "communication"],
    "Leadership": ["leadership", "team lead", "people management"],
    "Problem Solving": ["problem solving", "analytical skills"],
    "UI/UX Design": ["ui/ux", "ui design", "ux design", "figma", "adobe xd"],
    "Testing/QA": ["testing", "unit testing", "qa", "quality assurance", "selenium"],
    "Cybersecurity": ["cybersecurity", "information security", "penetration testing"],
    "Networking": ["networking", "tcp/ip", "network administration"],
    "Blockchain": ["blockchain", "solidity", "web3"],
}

MASTER_SKILLS: List[str] = sorted(SKILL_SYNONYMS.keys())

# 2. SYNTHETIC JOB POSTINGS DATASET (the "training data" for K-NN)
"""Each job has:
title
required_skills: dict of {skill: importance_weight (1-3)}
3 = must-have, 2 = important, 1 = nice-to-have
This structure lets us build both the K-NN feature vectors AND a weighted
skill-gap / readiness score."""

def get_job_dataset() -> List[dict]:
    return [
        {
            "title": "Data Scientist",
            "required_skills": {
                "Python": 3, "Machine Learning": 3, "Statistics": 3,
                "Pandas": 3, "NumPy": 2, "Scikit-learn": 3,
                "Data Visualization": 2, "SQL": 2, "Deep Learning": 1,
            },
        },
        {
            "title": "Machine Learning Engineer",
            "required_skills": {
                "Python": 3, "Machine Learning": 3, "Deep Learning": 3,
                "TensorFlow": 2, "PyTorch": 2, "Scikit-learn": 2,
                "SQL": 1, "System Design": 2, "Docker": 2,
            },
        },
        {
            "title": "Data Analyst",
            "required_skills": {
                "SQL": 3, "Excel": 3, "Data Analysis": 3,
                "Data Visualization": 3, "Statistics": 2, "Python": 2,
                "Communication": 2,
            },
        },
        {
            "title": "Backend Developer",
            "required_skills": {
                "Python": 2, "Java": 2, "REST API": 3, "SQL": 3,
                "Django": 1, "Flask": 1, "MySQL": 2, "MongoDB": 1,
                "Git": 2, "System Design": 2, "Microservices": 2,
            },
        },
        {
            "title": "Full Stack Developer",
            "required_skills": {
                "JavaScript": 3, "React": 3, "Node.js": 3, "HTML/CSS": 3,
                "REST API": 2, "MongoDB": 2, "SQL": 1, "Git": 2,
                "TypeScript": 1,
            },
        },
        {
            "title": "Frontend Developer",
            "required_skills": {
                "JavaScript": 3, "React": 3, "HTML/CSS": 3, "TypeScript": 2,
                "UI/UX Design": 2, "Git": 2, "Angular": 1,
            },
        },
        {
            "title": "DevOps Engineer",
            "required_skills": {
                "Linux": 3, "Docker": 3, "Kubernetes": 3, "CI/CD": 3,
                "AWS": 2, "Azure": 1, "Git": 2, "Networking": 2,
                "System Design": 1,
            },
        },
        {
            "title": "Cloud Engineer",
            "required_skills": {
                "AWS": 3, "Azure": 2, "GCP": 2, "Docker": 2,
                "Kubernetes": 2, "Networking": 2, "Linux": 2, "CI/CD": 1,
            },
        },
        {
            "title": "Data Engineer",
            "required_skills": {
                "Python": 3, "SQL": 3, "Spark": 2, "Hadoop": 1,
                "Big Data": 2, "AWS": 2, "Data Structures": 2,
                "Docker": 1, "PostgreSQL": 2,
            },
        },
        {
            "title": "QA / Test Engineer",
            "required_skills": {
                "Testing/QA": 3, "Python": 1, "Java": 1, "SQL": 1,
                "Agile/Scrum": 2, "Communication": 2,
            },
        },
        {
            "title": "Business Analyst",
            "required_skills": {
                "Data Analysis": 3, "Excel": 3, "SQL": 2,
                "Communication": 3, "Project Management": 2,
                "Data Visualization": 2, "Statistics": 1,
            },
        },
        {
            "title": "Project Manager",
            "required_skills": {
                "Project Management": 3, "Agile/Scrum": 3,
                "Communication": 3, "Leadership": 3, "Problem Solving": 2,
            },
        },
        {
            "title": "NLP Engineer",
            "required_skills": {
                "Python": 3, "NLP": 3, "Machine Learning": 2,
                "Deep Learning": 2, "PyTorch": 1, "TensorFlow": 1,
                "Statistics": 1,
            },
        },
        {
            "title": "Computer Vision Engineer",
            "required_skills": {
                "Python": 3, "Computer Vision": 3, "Deep Learning": 3,
                "PyTorch": 2, "TensorFlow": 2, "Machine Learning": 2,
            },
        },
        {
            "title": "Software Engineer (General)",
            "required_skills": {
                "Data Structures": 3, "Java": 2, "Python": 2,
                "SQL": 2, "Git": 2, "System Design": 2,
                "Problem Solving": 2, "REST API": 1,
            },
        },
        {
            "title": "Cybersecurity Analyst",
            "required_skills": {
                "Cybersecurity": 3, "Networking": 3, "Linux": 2,
                "Python": 1, "Problem Solving": 2,
            },
        },
        {
            "title": "UI/UX Designer",
            "required_skills": {
                "UI/UX Design": 3, "Communication": 2,
                "HTML/CSS": 1, "Problem Solving": 1,
            },
        },
        {
            "title": "Site Reliability Engineer (SRE)",
            "required_skills": {
                "Linux": 3, "Docker": 2, "Kubernetes": 3, "CI/CD": 2,
                "System Design": 2, "Networking": 2, "AWS": 2, "Python": 1,
            },
        },
        {
            "title": "Blockchain Developer",
            "required_skills": {
                "Blockchain": 3, "JavaScript": 2, "Python": 1,
                "System Design": 1, "Cybersecurity": 1,
            },
        },
        {
            "title": "Database Administrator",
            "required_skills": {
                "SQL": 3, "MySQL": 2, "PostgreSQL": 2, "MongoDB": 1,
                "Linux": 2, "System Design": 1,
            },
        },
    ]
