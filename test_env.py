from dotenv import load_dotenv
import os

load_dotenv()  # explicitly load .env in current directory

print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
