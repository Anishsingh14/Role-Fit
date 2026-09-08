import os
import sys

# Ensure the project root (one level up from tests/) is on sys.path so
# `import matcher_core`, `import job_data`, `import visualize` work
# regardless of the directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
