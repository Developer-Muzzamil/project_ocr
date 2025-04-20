
import os
import io
from google.cloud import vision

# Path to your Google Cloud service account key
key_path = "ocr-project-452822-494f171454e5.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Input and Output folder paths
input_folder = "200E"  # Folder containing images
output_folder = "output_texts"  # Folder to save OCR results

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

def process_image(image_path):
    """Extract text while preserving line structure using bounding box positions."""
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)  # Use document_text_detection for structured text

    if response.error.message:
        print(f"Error processing {image_path}: {response.error.message}")
        return None

    lines = []
    previous_y = None
    current_line = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    y_coord = word.bounding_box.vertices[0].y  # Get the Y coordinate

                    if previous_y is None:
                        previous_y = y_coord

                    # If Y coordinate difference is small, consider it the same line
                    if abs(y_coord - previous_y) > 10:  # Adjust threshold if needed
                        lines.append(" ".join(current_line))
                        current_line = [word_text]
                    else:
                        current_line.append(word_text)

                    previous_y = y_coord

    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)  # Keep original line breaks

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):  # Check for valid image formats
        image_path = os.path.join(input_folder, filename)
        text = process_image(image_path)

        if text:
            txt_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"Processed: {filename}")

print("🎉 OCR Extraction Completed! Check the 'output_texts' folder.")
