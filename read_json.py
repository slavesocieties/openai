import json
from layout_analysis_eval import calculate_metrics

def read_manual_annotations(path_to_json):
    """
    Reads coordinates of images that were manually annotated by CVAT
    
    Parameters
    ----------
    path_to_json : string
        a string representing the path to the json file
    
    Returns
    -------
    coord_dict_manual : dictionary
        a dictionary that contains the id and coordinates for the appropriate text boxes
    """
    
    manual_annotations = path_to_json

    with open(manual_annotations, "r") as file:
        data = json.load(file)

    coord_dict_manual = {}

    for item in data.get("images"):
        id_key = item["filename"].split(".")[0]
        coords = []
        for annotation in item.get("annotations", []):
            if annotation.get("name") == "text" and not annotation.get("attributes").get("margin"):
                coords.append(tuple(annotation["polygon"]))
        coord_dict_manual[id_key] = coords
    
    return coord_dict_manual

def read_computed_annotations(path_to_json):
    """
    Reads coordinates of images that were computationally annotated by segmentation driver
    
    Parameters
    ----------
    path_to_json : string
        a string representing the path to the json file
    
    Returns
    -------
    coord_dict_comp : dictionary
        a dictionary that contains the id and coordinates for all the computationally annotated text boxes
    """
    driver_annotations = path_to_json

    with open(driver_annotations, "r") as file:
        data = json.load(file)

    coord_dict_comp = {}

    for item in data:
        if item:
            id_key = item.get("image_id")
            coords = []
            for coord in item.get("original texts"):
                coords.append(tuple(coord["coords"]))
            coord_dict_comp[id_key] = coords

    return coord_dict_comp

def performance_metrics(path_to_manual, path_to_comp):
    """
    Main function that reads manually annotated coordinates and computationally annotated coordinates.
    Compares the two coordinates by evaluating each images coordinates with evaluation metrics precision, recall, and f-score.
    
    Parameters
    ----------
    path_to_manual : string
        a string representing the path to the json file for the manual annotations
    path_to_comp : string
        a string representing the path to the json file for the computed annotations

    Returns
    -------
    results : dictionary
        a dictionary that contains the id and evaluation metrics of each image
    """
    results = []
    coord_dict_manual = read_manual_annotations(path_to_manual)
    coord_dict_comp = read_computed_annotations(path_to_comp)
    for id in coord_dict_manual:
        entry = {}
        exists = False
        precision = recall = f_score = 0
        if id in coord_dict_comp:
            exists = True
            manual_coords = coord_dict_manual[id]
            comp_coords = coord_dict_comp[id]
            precision, recall, f_score = calculate_metrics(manual_coords, comp_coords)
    
        if exists:
            entry["image_id"] = id
            entry["evaluation metrics"] = []
            entry["evaluation metrics"].append({"precision": precision, "recall": recall, "f-score": f_score})
    
        else:
            entry["image_id"] = id
            entry["evaluation metrics"] = "N/A"
    
        results.append(entry)
    
    return results

with open("original_evaluation.json", "w") as f:
    json.dump(performance_metrics("copy of annotations.json", "full_size_images.json"), f, indent = 4)

