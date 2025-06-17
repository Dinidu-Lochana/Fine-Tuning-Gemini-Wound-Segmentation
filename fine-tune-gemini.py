from vertexai.preview.generative_models import GenerativeModel, TuningConfig
from vertexai.preview import generative_models

# Init Gemini model
model = GenerativeModel("gemini-1.5-flash-001")

# Define training dataset
tuned_model = model.tune_model(
    training_data="gs://gemini-segmentation/wound_dataset_groundingDino/annotations/wound_train.jsonl",
    model_display_name="wound-detection-gemini",
    tuning_config=TuningConfig(
        epoch_count=5,
        batch_size=4
    )
)

print("✅ Fine-tuning job started:", tuned_model.name)
