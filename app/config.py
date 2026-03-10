import os, logging
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(
    filename="api_usage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)