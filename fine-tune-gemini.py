from dotenv import load_dotenv
import os
import vertexai
from vertexai.tuning import sft


# Load environment variables from .env file
load_dotenv()

# Retrieve values
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")

# Initialize Vertex AI
vertexai.init(project=project_id, location=location)

sft_tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    train_dataset="gs://gemini-segmentation/wound_dataset_groundingDino/annotations/wound_train.jsonl"
)

# Poll for job completion
import time
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print("✅ Fine-tuning job started:", sft_tuning_job.tuned_model_name)
