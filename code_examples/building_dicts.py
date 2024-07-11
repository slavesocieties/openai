big_image_dict = {"images": [{"filename": "first_img.jpg"}, {"filename": "second_img.jpg"}]}

cropped_performance_data = {"images": [{"filename": "first_img.jpg", "cropped_perfomance": {}}]}

for image in cropped_performance_data:
    for index, im in enumerate(big_image_dict):
        if image["filename"] == im["filename"]:
            big_image_dict[index]["cropped_performance"] = image["cropped_performance"]