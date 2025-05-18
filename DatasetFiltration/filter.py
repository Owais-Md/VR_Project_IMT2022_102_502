import os
import gzip
import json
import pandas as pd
from glob import glob
from collections import defaultdict
import random
import shutil
from pathlib import Path

# Part 1: Filter and curate listings
def curate_listings():
    # Load image metadata into a lookup dictionary
    image_data = pd.read_csv("./dataset/images/metadata/images.csv")
    image_to_path = dict(zip(image_data["image_id"], image_data["path"]))

    # Define target languages
    valid_langs = {"en_IN", "en_US", "en_CA", "en_GB", "en_SG", "en_AU"}

    # Specify metadata fields to extract
    target_fields = [
        "bullet_point", "color", "color_code", "fabric_type", "item_name", 
        "item_shape", "material", "pattern", "product_description", 
        "product_type", "style"
    ]

    # Find all listing files
    listing_files = sorted(glob("./dataset/listings/metadata/listings_*.json*"))

    # Process listings and write to output
    with open("./dataset_curated/curated_listings.jsonl", "w", encoding="utf-8") as output_file:
        for file in listing_files:
            opener = gzip.open if file.endswith(".gz") else open
            with opener(file, "rt", encoding="utf-8") as input_file:
                for line in input_file:
                    try:
                        record = json.loads(line)

                        # Determine language from key fields
                        detected_lang = None
                        for field in ["item_name", "item_keywords", "bullet_point"]:
                            if field in record:
                                for item in record[field]:
                                    if isinstance(item, dict) and item.get("language_tag") in valid_langs:
                                        detected_lang = item["language_tag"]
                                        break
                            if detected_lang:
                                break

                        if not detected_lang:
                            continue

                        # Validate image ID and path
                        img_id = record.get("main_image_id")
                        if not img_id or img_id not in image_to_path:
                            continue

                        # Create output entry
                        output_entry = {"image_path": image_to_path[img_id]}

                        # Extract desired fields
                        for field in target_fields:
                            value = record.get(field)
                            if isinstance(value, list):
                                filtered_values = [
                                    item["value"] for item in value
                                    if isinstance(item, dict) and item.get("language_tag") == detected_lang and "value" in item
                                ]
                                if filtered_values:
                                    output_entry[field] = filtered_values
                            elif isinstance(value, (str, list)):
                                output_entry[field] = value

                        # Write entry to file
                        json.dump(output_entry, output_file)
                        output_file.write("\n")

                    except Exception:
                        continue

# Part 2: Check for duplicate image paths
def check_duplicates():
    unique_paths = set()
    repeated_paths = set()

    with open("./dataset_curated/curated_listings.jsonl", "r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            path = record.get("image_path")
            if path:
                if path in unique_paths:
                    repeated_paths.add(path)
                else:
                    unique_paths.add(path)

    print(f"Count of repeated image paths: {len(repeated_paths)}")
    print("Repeated image paths:")
    for path in repeated_paths:
        print(path)

# Part 3: Remove duplicates, keeping entry with most attributes
def remove_duplicates():
    input_file = "./dataset_curated/curated_listings.jsonl"
    output_file = "./dataset_curated/unique_listings.jsonl"

    path_to_entry = {}

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                path = record.get("image_path")
                if not path:
                    continue

                # Keep entry with more attributes
                if path not in path_to_entry or len(record) > len(path_to_entry[path]):
                    path_to_entry[path] = record
            except json.JSONDecodeError:
                print("Skipping malformed JSON line")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for record in path_to_entry.values():
            json.dump(record, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Reduced to {len(path_to_entry)} unique entries by image path.")

# Part 4: Create disjoint splits with balanced product types
def create_splits():
    # Configuration
    split_names = ["S1", "S2", "S3", "S4", "S5", "S6"]
    split_size = 3000
    expected_product_types = 576
    image_root_dir = "./dataset/images/small/"
    output_root_dir = "./dataset_curated"

    # Load unique listings
    listings = []
    with open("./dataset_curated/unique_listings.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            listings.append(json.loads(line))

    # Group by product type
    product_type_groups = defaultdict(list)
    for entry in listings:
        product_type = entry.get("product_type")
        if product_type:
            product_type_groups[product_type].append(entry)

    # Initialize splits
    splits = {name: [] for name in split_names}
    used_indices = set()

    # Create each split
    for split_name in split_names:
        current_split = []

        # Ensure one entry per product type
        for product_type, entries in product_type_groups.items():
            selected_entry = random.choice(entries)
            current_split.append(selected_entry)
            used_indices.add(listings.index(selected_entry))

        # Fill remaining slots with random entries
        available_entries = [e for i, e in enumerate(listings) if i not in used_indices]
        required_count = split_size - len(current_split)
        selected_entries = random.sample(available_entries, required_count)
        current_split.extend(selected_entries)
        used_indices.update(listings.index(e) for e in selected_entries)
        splits[split_name] = current_split

    # Write splits to disk
    for split_name, entries in splits.items():
        split_folder = os.path.join(output_root_dir, split_name)
        img_folder = os.path.join(split_folder, f"{split_name}_images")
        meta_folder = os.path.join(split_folder, f"{split_name}_metadata")

        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(meta_folder, exist_ok=True)

        # Process metadata and copy images
        metadata_entries = []
        for entry in entries:
            src_path = entry["image_path"]
            file_name = os.path.basename(src_path)
            dest_image_path = os.path.join(img_folder, file_name)
            full_src_path = os.path.join(image_root_dir, src_path)

            # Update metadata
            entry["image_path"] = str(dest_image_path)
            metadata_entries.append(entry)

            # Copy image
            try:
                shutil.copy(full_src_path, dest_image_path)
            except Exception as e:
                print(f"Error copying {full_src_path}: {e}")

        # Save metadata
        with open(os.path.join(meta_folder, f"{split_name}_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_entries, f, indent=2, ensure_ascii=False)

# Execute all parts
if __name__ == "__main__":
    curate_listings()
    check_duplicates()
    remove_duplicates()
    create_splits()