# VR Mini Project: Part 2

## Submission
----------
- **Contributors** :
  - Aryan Mishra(IMT2022502, Aryan.Mishra@iiitb.ac.in)
  - Md Owais(IMT2022102, Mohammad.Owais@iiitb.ac.in)
- **Clone GitHub Repository** :
  ```
  https://github.com/Owais-Md/VR_Project_IMT2022_102_502.git
  ```
 
## Folder Structure

- **Blip2/**  
  - `blip_vqa_predictions.csv` – Ground-truth vs. predicted answers.  
  - `blip_vqa_metrics.csv` – Computed metrics (Accuracy, F1, BERTScore) based on the predictions.  
  - `notebook.ipynb` – Single notebook that runs inference and saves both CSVs.  
  - `requirements.txt` – Dependencies to reproduce this analysis.

- **ViLT/**  
  - `baseline.ipynb` – Evaluation of the pre-trained ViLT model.  
  - `finetuned.ipynb` – Evaluation of the LoRA-fine-tuned ViLT model.  
  - `requirements.txt` – Environment specs for both notebooks.

- **blipvqa/**  
  - `requirements.txt` – Shared dependencies for all subfolders.  
  - `Blip-vqa Baseline Code.ipynb` – Runs and logs baseline BLIP VQA Base evaluation.  
  - `evaluating-finetuned-model.ipynb` – Evaluates the LoRA-fine-tuned BLIP VQA Base.  
  - **Subfolders** (`experiment_1/`, `experiment_2/`, …) each contain:  
    - `notebook.ipynb` – Model training/fine-tuning for that experiment.  
    - `results/` – Output folder with saved model weights (from `trainer.save_model()`) and logs.

- **DataFiltration/**  
  - `filter.py` – Splits the raw ABO dataset into subfolders `S1`–`S6`.  
  - `curate.py` – Generates VQA QA-pair CSVs for a given subfolder using a Gemini API key.  
  - *Note:* Both scripts use `requirements.txt` from the root.

- **IMT2022102_IMT2022502_inference_script/**  
  - `inference.py` – Final inference script as per project deliverables.  
  - `requirements.txt` – Dependencies for running inference.

- **Dataset.csv**  
  - The merged, trimmed CSV of subfolders `S1`–`S6`, containing the final VQA pairs used throughout.

- **VR_project_report.pdf**  
  - The complete project report, including methodology, results, and analysis.
