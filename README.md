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

Folder: Blip 2 has: a single ipynb, and 2 csvs, blip_vqa_predictions.csv with the ground truths and predicted values, and blip_vqa_metrics.csv with the metrics calculated on the previous csv
and has requirments.txt

Folder: ViLT has: 2 python notebooks for both baseline and finetuned models(results can  be seen in the notebook), and has requirments.txt

Folder: blipvqa baseline and finetuned: one requirments.txt, Blip-vqa Baseline Code.ipynb, which does evaluation of baseline blip-vqa model and stores results in the ipynb itself, and evaluating-finetuned-model.ipynb, which evaluates the finetuned model.

The subfolders in blipvqa, each have one ipynb and a results folder, where the ipynb dependencies are the same as the requirements.txt folder outside. Running the notebooks gives the output result folder, which has the finetuned model weights and parameters from the trainer.save_model().

Folder: DataFiltration has filter.py and curate.py which have requirements from the requirement.txt
running filter.py in a folder with the extracted dataset, will create the subfolders S1-S6 as mentioned in the report, and running curate.py with a Gemini API key and subfolder name specified generates the required vqa pairs csv.

Folder: IMT2022102_IMT2022502_inference_script has the inference.py and requirements.txt as asked in project deliverables.

Dataset.csv is the combined trim of the subfolders S1 to S6, which is basically the final dataset of VQA Pairs that was used for the entire project.

VR_project_report.pdf is the detailed report for the project.
