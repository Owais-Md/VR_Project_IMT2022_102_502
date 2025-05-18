import os
import json
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import time

# Configuration
DATA_ROOT = './dataset_curated'
SPLIT_NAME = 'S1'
METADATA_PATH = os.path.join(DATA_ROOT, SPLIT_NAME, f'{SPLIT_NAME}_metadata')
IMAGE_PATH = os.path.join(DATA_ROOT, SPLIT_NAME, f'{SPLIT_NAME}_images')
OUTPUT_CSV = os.path.join(DATA_ROOT, SPLIT_NAME, f'{SPLIT_NAME}_qa_pairs.csv')

# Initialize Gemini AI
genai.configure(api_key="...")
ai_model = genai.GenerativeModel('models/gemini-1.5-flash')

# Load previously processed images
completed_images = set()
if os.path.exists(OUTPUT_CSV):
    try:
        existing_data = pd.read_csv(OUTPUT_CSV, header=None, names=['image_path', 'question', 'answer'])
        completed_images = set(existing_data['image_path'].unique())
        print(f"Found {len(completed_images)} previously processed images.")
    except pd.errors.EmptyDataError:
        print("CSV file is empty. Starting from scratch.")
        completed_images = set()

# Collect unprocessed metadata
pending_entries = []
for file_name in os.listdir(METADATA_PATH):
    if file_name.endswith('.json'):
        with open(os.path.join(METADATA_PATH, file_name), 'r', encoding='utf-8') as file:
            data = json.load(file)
            for record in data:
                img_path = record.get('image_path')
                if not img_path or not os.path.exists(img_path):
                    continue
                if img_path in completed_images:
                    continue

                # Convert metadata fields to a single string
                metadata_parts = []
                for key, value in record.items():
                    if key == "image_path":
                        continue
                    if isinstance(value, list):
                        value = ', '.join(str(item) for item in value)
                    elif not isinstance(value, str):
                        continue
                    if value.strip():
                        metadata_parts.append(f"{key}: {value.strip()}")

                metadata_text = ', '.join(metadata_parts)
                pending_entries.append((img_path, metadata_text))

print(f"Identified {len(pending_entries)} new images for processing.")

# Define prompt template
QA_PROMPT = """Using the product image and this metadata: {metadata},
create 5 concise question-answer pairs to assess visual comprehension.
Each answer must be one word (e.g., 'Blue', 'Shirt', 'Wood').
Format as: Q1: <question> A1: <answer> ..."""

# Process images and generate QA pairs
results = []

for img_path, metadata in tqdm(pending_entries):
    try:
        with open(img_path, 'rb') as img_file:
            img_content = img_file.read()

        formatted_prompt = QA_PROMPT.format(metadata=metadata)

        # Generate response
        ai_response = ai_model.generate_content([
            formatted_prompt,
            {"mime_type": "image/jpeg", "data": img_content}
        ])

        time.sleep(4.3)  # Adhere to rate limit (~14 requests/min)

        if not ai_response.text:
            print(f"[NO RESPONSE] {img_path}")
            continue

        response_lines = ai_response.text.strip().split('\n')

        # Parse QA pairs
        for line in response_lines:
            if line.startswith("Q") and "A" in line:
                try:
                    q_part = line.split("A")[0].split(":")[1].strip()
                    a_part = line.split("A")[1].split(":")[1].strip()
                    results.append((img_path, q_part, a_part))
                except:
                    continue

        # Save results incrementally
        if results:
            result_df = pd.DataFrame(results, columns=['image_path', 'question', 'answer'])
            result_df.to_csv(OUTPUT_CSV, mode='a', header=None, index=False)
            results.clear()

    except Exception as error:
        if 'quota' in str(error).lower():
            print(f"[QUOTA LIMIT] Processing stopped at {img_path}")
            break
        print(f"[ERROR] {img_path}: {str(error)}")
        continue

print("Processing completed.")