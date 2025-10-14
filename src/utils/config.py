from dotenv import load_dotenv
import os

load_dotenv(override=True)


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT environment variable not set")
