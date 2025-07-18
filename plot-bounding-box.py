import cv2
import os

# Properly formatted image path (note: raw string `r""` to avoid escape issues)
image_path = r"C:\24\Kainovation Technologies\Fine-Tuning-Gemini-Wound-Segmentation\wound_dataset_groundingDino\images\train\4c08ba__jpg.rf.0e3ab0e62155c14583526cf6e5daeb87.jpg"

# Check if the image exists
if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    image = cv2.imread(image_path)

    response = {
        "wounds": [
            {"label": "0", "box_2d":[48, 153, 172.326, 269.822]},
            {"label": "0", "box_2d":[442, 136, 553.54, 245.281]},
           
             # [248, 414, 395.993, 591.789]
            
        ]
    }

    for wound in response["wounds"]:
        x1, y1, x2, y2 = map(int, wound["box_2d"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"Label: {wound['label']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Wound Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
