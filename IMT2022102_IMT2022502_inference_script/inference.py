import argparse
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import accelerate
import os

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on image dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to image-metadata CSV")
    return parser.parse_args()

def load_data(csv_path):
    """Load metadata from CSV file."""
    return pd.read_csv(csv_path)

def setup_device():
    """Set up the accelerator and return the device."""
    accelerator = accelerate.Accelerator()
    return accelerator.device

def load_model_and_processor(model_path, device):
    """Load the processor and model from the specified path and move the model to the device."""
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForQuestionAnswering.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return processor, model

def generate_answer(model, processor, image_path, question, device):
    """Generate an answer for the given image and question."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32)
            answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            answer = answer.split()[0].lower()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        answer = "error"
    return answer

def save_results(df, output_path="results.csv"):
    """Save the dataframe with generated answers to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    """Main function to run the inference process."""
    args = parse_args()
    df = load_data(args.csv_path)
    device = setup_device()
    # https://huggingface.co/owais-md/blip-vqa-16-3
    processor, model = load_model_and_processor("owais-md/blip-vqa-16-3", device)
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        image_path = os.path.join(args.image_dir, row["image_name"])
        question = str(row["question"])
        answer = generate_answer(model, processor, image_path, question, device)
        generated_answers.append(answer)
    df["generated_answer"] = generated_answers
    save_results(df)

if __name__ == "__main__":
    main()