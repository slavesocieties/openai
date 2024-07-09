import json
from PIL import Image, ImageDraw
import os
import shutil

def draw_rectangles(image, annotations, color):
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["polygon"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image

def flip_image_and_annotations(input_json, output_json, image_dir, output_image_dir):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input JSON: {os.path.abspath(input_json)}")
    print(f"Image directory: {os.path.abspath(image_dir)}")
    print(f"Output image directory: {os.path.abspath(output_image_dir)}")
    print(f"Output JSON file: {os.path.abspath(output_json)}")

    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    os.makedirs(output_image_dir)
    os.makedirs(os.path.join(output_image_dir, "verification"))

    with open(input_json, 'r') as f:
        data = json.load(f)

    flipped_data = {"images": []}

    for img_data in data["images"]:
        img_path = os.path.join(image_dir, img_data["filename"])
        print(f"Processing image: {img_path}")

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue

        try:
            img = Image.open(img_path)
            original_with_rect = draw_rectangles(img.copy(), img_data["annotations"], "red")

            # Create all flipped versions
            h_flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            v_flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            hv_flipped_img = img.transpose(Image.ROTATE_180)

            # Save all flipped versions
            flipped_versions = {
                "h_flipped": h_flipped_img,
                "v_flipped": v_flipped_img,
                "hv_flipped": hv_flipped_img
            }

            for prefix, flipped_img in flipped_versions.items():
                flipped_filename = f"{prefix}_{img_data['filename']}"
                flipped_img_path = os.path.join(output_image_dir, flipped_filename)
                flipped_img.save(flipped_img_path)

            image_width = img_data["size"]["width"]
            image_height = img_data["size"]["height"]

            flipped_annotations = {
                "h_flipped": [],
                "v_flipped": [],
                "hv_flipped": []
            }

            for annotation in img_data["annotations"]:
                x1, y1, x2, y2 = annotation["polygon"]

                # Horizontal flip annotations
                h_flipped_annotation = annotation.copy()
                h_flipped_annotation["polygon"] = [image_width - x2, y1, image_width - x1, y2]
                flipped_annotations["h_flipped"].append(h_flipped_annotation)

                # Vertical flip annotations
                v_flipped_annotation = annotation.copy()
                v_flipped_annotation["polygon"] = [x1, image_height - y2, x2, image_height - y1]
                flipped_annotations["v_flipped"].append(v_flipped_annotation)

                # Both horizontal and vertical flip annotations
                hv_flipped_annotation = annotation.copy()
                hv_flipped_annotation["polygon"] = [image_width - x2, image_height - y2, image_width - x1, image_height - y1]
                flipped_annotations["hv_flipped"].append(hv_flipped_annotation)

            # Draw rectangles on flipped images
            colors = {"h_flipped": "blue", "v_flipped": "green", "hv_flipped": "purple"}
            flipped_with_rect = {}
            for flip_type, annotations in flipped_annotations.items():
                flipped_with_rect[flip_type] = draw_rectangles(flipped_versions[flip_type].copy(), annotations, colors[flip_type])

            # Save verification images
            verification_path = os.path.join(output_image_dir, "verification", f"verify_{img_data['filename']}")
            original_with_rect.save(verification_path.replace(".jpg", "_original.jpg"))
            for flip_type, img_with_rect in flipped_with_rect.items():
                img_with_rect.save(verification_path.replace(".jpg", f"_{flip_type}.jpg"))

            # Add all flipped versions to the output data
            for flip_type in flipped_versions.keys():
                flipped_data["images"].append({
                    "filename": f"{flip_type}_{img_data['filename']}",
                    "size": img_data["size"],
                    "annotations": flipped_annotations[flip_type]
                })

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

    with open(output_json, 'w') as f:
        json.dump(flipped_data, f, indent=2)

    print(f"Processed {len(data['images'])} original images, created {len(flipped_data['images'])} flipped images.")
    print(f"Output saved to {output_json}")
    print(f"Verification images saved in {os.path.join(output_image_dir, 'verification')}")

# File paths
input_json = "annotations.json"
output_json = "flipped_annotations.json"
image_dir = "original_pics"
output_image_dir = "new_pics"

flip_image_and_annotations(input_json, output_json, image_dir, output_image_dir)
