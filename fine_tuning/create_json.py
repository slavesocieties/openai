import json
import xml.etree.ElementTree as ET

def parse_xml_transcriptions(xml_file):
    """Parse the XML file and return a dictionary of transcriptions indexed by their transformed ID."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    transcriptions = {}
    
    for image in root.findall(".//image"):
        image_id = f"0{image.get('id')[1:]}"  # Transform 1xxx -> 0xxx
        for entry in image.findall("entry"):
            entry_id = f"{int(entry.get('id')):02d}"  # Pad entry ID to two digits
            lines = entry.text.split("\n") if entry.text else []
            filtered_lines = [line.strip() for line in lines if len(line.strip()) >= 20]
            
            transcriptions[f"{image_id}-{entry_id}"] = " ".join(filtered_lines)
    
    return transcriptions

def process_transcriptions(annotation_file, xml_file, output_json):
    """Process annotation and XML files to produce a structured JSON output."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    transcriptions = parse_xml_transcriptions(xml_file)
    output_entries = []
    
    for image_info in data["images"]:
        annotations = image_info["annotations"]
        
        # Filter relevant text annotations (non-partial, non-margin)
        text_annotations = [
            anno for anno in annotations 
            if anno["name"] == "text" and not anno["attributes"]["partial"] and not anno["attributes"]["margin"]
        ]
        
        # Sort annotations by vertical position (top coordinate)
        text_annotations.sort(key=lambda x: x["polygon"][1])
        
        # Match transcriptions to text annotations
        for anno in text_annotations:
            annotation_id = f"0{image_info['filename'][1:4]}-{int(anno['polygon'][1]):02d}"  # Generate matching ID
            if annotation_id in transcriptions:
                output_entries.append({
                    "id": annotation_id,
                    "lines": transcriptions[annotation_id]
                })
    
    # Save structured JSON output
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({"entries": output_entries}, f, ensure_ascii=False, indent=4)
    
    print(f"Processed transcriptions saved to {output_json}")

# Example usage
annotation_file = "annotations.json"
xml_file = "239746.xml"
output_json = "processed_transcriptions.json"
process_transcriptions(annotation_file, xml_file, output_json)