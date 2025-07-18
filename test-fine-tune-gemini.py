from vertexai.preview.generative_models import GenerativeModel, Part

# Load deployed model
model = GenerativeModel("projects/infra-ratio-462407-t9/locations/us-east1/endpoints/7968446639515893760")

# Create image part from GCS URI
image_part = Part.from_uri(
    uri="gs://gemini-segmentation/wound_dataset_groundingDino/images/train/4c08ba__jpg.rf.0e3ab0e62155c14583526cf6e5daeb87.jpg",
    mime_type="image/jpeg"  # Adjust if using PNG/WEBP
)

text_part = Part.from_text("Give the wounds cordinates as [xmin,ymin,xmax,ymax]")

response = model.generate_content(
    contents=[text_part, image_part]
)

print("🧠 Gemini says:", response.text)
