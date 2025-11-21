from dotenv import load_dotenv
import os

load_dotenv(override=True)


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT environment variable not set")


EVALUATION_RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "storage",
    "evaluation_results"
)

TRAINED_MODELS_PATH = os.path.join(
    PROJECT_ROOT,
    "storage",
    "models"
)
