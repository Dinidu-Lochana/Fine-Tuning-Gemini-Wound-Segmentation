import json
import os

# Update these paths to match your local file structure
image_folder_path = r"C:\24\Kainovation Technologies\Fine-Tuning-Gemini-Wound-Segmentation\wound_dataset_groundingDino\images\train"
coco_path = r"C:\24\Kainovation Technologies\Fine-Tuning-Gemini-Wound-Segmentation\wound_dataset_groundingDino\annotations\train.json"
output_jsonl = r"C:\24\Kainovation Technologies\Fine-Tuning-Gemini-Wound-Segmentation\wound_dataset_groundingDino\annotations\wound_train.jsonl"

# Load the COCO-style annotation file
with open(coco_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# Maps: image ID → filename and category ID → name
image_map = {img['id']: img['file_name'] for img in coco_data['images']}
category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Build image → label set
img_labels = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    label = category_map.get(ann['category_id'], "unknown")
    if img_id not in img_labels:
        img_labels[img_id] = set()
    img_labels[img_id].add(label)

# Write JSONL file with local file paths
with open(output_jsonl, 'w', encoding='utf-8') as out_file:
    for img_id, labels in img_labels.items():
        filename = image_map.get(img_id)
        if not filename:
            continue
        # Full local image path (Windows-safe)
        local_image_path = os.path.abspath(os.path.join(image_folder_path, filename))
        caption = "This image contains: " + ", ".join(sorted(labels)) + "."
        json_line = {
            "input": {"image": local_image_path},
            "output": caption
        }
        out_file.write(json.dumps(json_line) + "\n")

print("✅ Local Gemini JSONL file created at:", output_jsonl)
