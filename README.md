# VR Mini Project: Part 2

## Submission
----------
- **Contributors** :
  - Aryan Mishra(IMT2022502, Aryan.Mishra@iiitb.ac.in)
  - Md Owais(IMT2022102, Mohammad.Owais@iiitb.ac.in)
- **Clone GitHub Repository** :
  ```
  https://github.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502.git
  ```
- Files:
  - **BinaryClassification.ipynb** : Tasks A and B.
    - **all_cnn_hyperparameters.csv** : All the hyperparameters used and their results in Task B.
    - **best_cnn_hyperparameters.csv** : The hyperparameters which gave the best results in Task B.
  - **Segmentation.ipynb** : Tasks C and D.
  - **README.md** : This file.
 
## Introduction
------------
This project focuses on developing a computer vision solution to classify and segment face masks in images, addressing a critical need for automated detection systems in contexts like public health monitoring. The objective is to implement and compare two approaches:

- **Binary Classification** : Determining whether a person in an image is wearing a face mask ("with mask") or not ("without mask") using:
  - Handcrafted features with traditional machine learning (ML) classifiers.
  - A Convolutional Neural Network (CNN).
- **Mask Segmentation** : Identifying and delineating the mask region in images of people wearing masks using:
  - Traditional region-based segmentation techniques.
  - A U-Net deep learning model.

The implementation leverages Python, utilizing libraries such as OpenCV, scikit-learn, TensorFlow, and PyTorch, to process images, train models, and evaluate performance.

## Dataset
-------
### Sources

The project utilizes two publicly available datasets:

#### Face Mask Detection Dataset:
- **Source** : https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
- **Description** : Contains images of people with and without face masks, labeled for binary classification tasks.
- **Structure** :
```
  dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```
- **Access** : We used the zip file('dataset.zip') from the above link as our dataset.
        
#### Masked Face Segmentation Dataset (MFSD):
- **Source** : https://github.com/sadjadrz/MFSD
- **Description** : Provides images with corresponding ground truth segmentation masks for faces with masks.
- **Structure** :
```
  MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img
```
- **Access** : We used the zip file('MSFD.zip') from the above link as our dataset.
- **Note** : The dataset can be accessed using the link above and hasn't been downloaded as they are they are very large in size.

## Preprocessing
-------------
- **Classification Dataset** : Images are resized to 64x64 pixels, normalized, and split into training (80%) and validation (20%) sets.
- **Segmentation Dataset** : Images and masks are resized to 128x128 pixels, normalized, and split into training (80%) and validation (20%) sets for U-Net training.

## Methodology
-----------
### Task A: Binary Classification Using Handcrafted Features and ML Classifiers
-------------------------------------------------------------------------------
#### A.i: Extract Handcrafted Features

- **Features** : Both Histogram of Oriented Gradients (HOG) features and Scale Invariant Feature Transform(SIFT) features are extracted from the Face Mask Detection dataset and as HOG gave better results, that has been used to train and evaluate the ML classifiers.
  - HOG:
    - ![Screenshot 2025-03-25 at 6 02 17 PM](https://github.com/user-attachments/assets/3f100aa1-12c5-4f37-8850-6237f877b19c)
    - A HOG feature vector of length 1764 means that each image is represented by 1764 numbers(block-normalised histogram bin values) that capture its gradient orientations and edge information.
  - SIFT:
    - ![Screenshot 2025-03-25 at 6 07 55 PM](https://github.com/user-attachments/assets/902a168f-0a1d-4cfd-9822-e5f0e07dadcd)
    - The shape (4095, 128) indicates that SIFT found 4095 keypoints, and each keypoint is described by a 128-dimensional SIFT descriptor.

- **Process** : Images are loaded from finaldataset.zip, resized to 64x64, and converted to grayscale before both HOG and SIFT feature extraction.

#### A.ii: Train and Evaluate ML Classifiers

- **Classifiers** :
  - **Support Vector Machine (SVM)** : We tried altering between various kernels and finally used 'rbf' kernel as it gave the best results.
  - **Neural Network** : To get the best results, we had to alter in various ways like adding dropout layers, using adam optimiser, experimenting with the number of hidden layers and the number of nodes in each layer to get the best result possible. The final structure is as shown below:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_34 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">451,840</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">65</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>


- **XGBoost** : To get the best results, we create DMatrices(optimised data structure for XGBoost) using the test and train data, experimented between the parameters and used xgb.cv(k fold cross-validation) with early stopping. We got 86 rounds as the best number of rounds for cross validation and used the following parameters: 

   ![Screenshot 2025-03-25 at 4 43 48 PM](https://github.com/user-attachments/assets/7608228b-a2e0-4ff9-96a3-766eb3b8bbc7)


   ![Screenshot 2025-03-25 at 4 44 33 PM](https://github.com/user-attachments/assets/9231294f-5641-431d-b7d3-4162e8c66b34)
   
- **Training**: Features are split into training and validation sets (80-20 split), and classifiers are trained using sklearn.
- **Evaluation**: Accuracy is computed on the validation set using accuracy_score.

#### A.iii: Report and Compare Accuracy

- Results are compared between SVM, Neural Network and XGBoost based on validation accuracy.

### Task B: Binary Classification Using CNN
--------------------------------------------------
#### B.i: Design and Train a CNN

- **Architecture** : We used 3 convolutional blocks followed by fully connected layers followed by the output layer for binary classification(Model: 'sequential_11').
  - Conv layers: extract hierarchial features.
  - Pooling layers(2x2 Max Pooling): reduce spatial dimensions.
  - Activation function: ReLU(introduces non-linearity).
  - Fully connected layers(Dense(128, activation='relu')): Fully connected hidden layers(like dense layers in traditional Neural network)
  - Dropout rate: Prevents overfitting
  - Filters: Increasing filters to caputre more complex features.
  - Output layerDense((1, activation='sigmoid')): Binary classification(mask/no mask) output
- The structure can be seen below:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">62</span>, <span style="color: #00af00; text-decoration-color: #00af00">62</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">31</span>, <span style="color: #00af00; text-decoration-color: #00af00">31</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">29</span>, <span style="color: #00af00; text-decoration-color: #00af00">29</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4608</span>)                │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">589,952</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">129</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>

- **Training** : Images are loaded via a custom ZipDataGenerator, features are split into training and validation sets (80-20 split), trained with Adam optimizer and binary cross-entropy loss.

#### B.ii: Hyperparameter Variations

- **Variations Tested** :
  - Learning rates: 0.01, 0.001, 0.0001.
  - Optimizers: Adam, SGD, RMSprop.
  - Batch sizes: 16, 32, 64.
  - Dropout rates: 0.3, 0.4, 0.5.
  - Final activation: Sigmoid, ReLU, Softmax.
- **Process**: Models are trained for 5 epochs with early stopping (patience=2) and evaluated on validation accuracy.
- (The hyperparameters used and their results are shown in the section below.)

#### B.iii: Compare CNN with ML Classifiers

- The best CNN configuration’s accuracy is compared against the traditional ML classifiers(SVM, Neural Network and XGBoost) results.

### Task C: Region Segmentation Using Traditional Techniques
--------------------------------------------------------------------
#### C.i: Implement Region-Based Segmentation

- **Otsu's Thresholding**: 
    - Load images from MSFD.zip (face_crop).
    - Use Open CV's inbuilt otsu-thresholding.
    - Identify mask and refine with binary closing.
    - Generate binary mask.
- **Region Growing with Flood Fill:**
    - Seed point initialized at the center of the image.
    - Flood fill applied with a tolerance to capture nearby pixels of similar intensity.
    - Morphological closing applied to refine the segmentation result.

#### C.ii: Visualize and Evaluate

- **Visualization** : Input image, ground truth, and predicted mask are plotted.
- **Metrics** : Intersection over Union (IoU) and Dice score are computed against ground truth masks from face_crop_segmentation.

### Task D: Mask Segmentation Using U-Net
-----------------------------------------------
#### D.i: Train a U-Net Model

- **Architecture** :
  - Encoder: 4 downsampling blocks (Conv2D + ReLU + MaxPooling).
  - Bottleneck: Conv2D block at the deepest level.
  - Decoder: 4 upsampling blocks (UpSampling2D + Conv2D + ReLU) with skip connections to preserve spatial information.
  - Output: 1x1 Conv2D with Sigmoid activation.
``` 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                   ┃ Output Shape                 ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Conv2d                         │ [1, 64, 256, 256]            │            1792 │
│ ReLU                           │ [1, 64, 256, 256]            │               0 │
│ Conv2d                         │ [1, 64, 256, 256]            │           36928 │
│ ReLU                           │ [1, 64, 256, 256]            │               0 │
│ Sequential                     │ [1, 64, 256, 256]            │           38720 │
│ DoubleConv                     │ [1, 64, 256, 256]            │           38720 │
│ MaxPool2d                      │ [1, 64, 128, 128]            │               0 │
│ EncoderBlock                   │ [1, 64, 256, 256]            │           38720 │
│ EncoderBlock                   │ [1, 64, 128, 128]            │           38720 │
│ Conv2d                         │ [1, 128, 128, 128]           │           73856 │
│ ReLU                           │ [1, 128, 128, 128]           │               0 │
│ Conv2d                         │ [1, 128, 128, 128]           │          147584 │
│ ReLU                           │ [1, 128, 128, 128]           │               0 │
│ Sequential                     │ [1, 128, 128, 128]           │          221440 │
│ DoubleConv                     │ [1, 128, 128, 128]           │          221440 │
│ MaxPool2d                      │ [1, 128, 64, 64]             │               0 │
│ EncoderBlock                   │ [1, 128, 128, 128]           │          221440 │
│ EncoderBlock                   │ [1, 128, 64, 64]             │          221440 │
│ Conv2d                         │ [1, 256, 64, 64]             │          295168 │
│ ReLU                           │ [1, 256, 64, 64]             │               0 │
│ Conv2d                         │ [1, 256, 64, 64]             │          590080 │
│ ReLU                           │ [1, 256, 64, 64]             │               0 │
│ Sequential                     │ [1, 256, 64, 64]             │          885248 │
│ DoubleConv                     │ [1, 256, 64, 64]             │          885248 │
│ MaxPool2d                      │ [1, 256, 32, 32]             │               0 │
│ EncoderBlock                   │ [1, 256, 64, 64]             │          885248 │
│ EncoderBlock                   │ [1, 256, 32, 32]             │          885248 │
│ Conv2d                         │ [1, 512, 32, 32]             │         1180160 │
│ ReLU                           │ [1, 512, 32, 32]             │               0 │
│ Conv2d                         │ [1, 512, 32, 32]             │         2359808 │
│ ReLU                           │ [1, 512, 32, 32]             │               0 │
│ Sequential                     │ [1, 512, 32, 32]             │         3539968 │
│ DoubleConv                     │ [1, 512, 32, 32]             │         3539968 │
│ MaxPool2d                      │ [1, 512, 16, 16]             │               0 │
│ EncoderBlock                   │ [1, 512, 32, 32]             │         3539968 │
│ EncoderBlock                   │ [1, 512, 16, 16]             │         3539968 │
│ Conv2d                         │ [1, 1024, 16, 16]            │         4719616 │
│ ReLU                           │ [1, 1024, 16, 16]            │               0 │
│ Conv2d                         │ [1, 1024, 16, 16]            │         9438208 │
│ ReLU                           │ [1, 1024, 16, 16]            │               0 │
│ Sequential                     │ [1, 1024, 16, 16]            │        14157824 │
│ DoubleConv                     │ [1, 1024, 16, 16]            │        14157824 │
│ ConvTranspose2d                │ [1, 512, 32, 32]             │         2097664 │
│ Conv2d                         │ [1, 512, 32, 32]             │         4719104 │
│ ReLU                           │ [1, 512, 32, 32]             │               0 │
│ Conv2d                         │ [1, 512, 32, 32]             │         2359808 │
│ ReLU                           │ [1, 512, 32, 32]             │               0 │
│ Sequential                     │ [1, 512, 32, 32]             │         7078912 │
│ DoubleConv                     │ [1, 512, 32, 32]             │         7078912 │
│ DecoderBlock                   │ [1, 512, 32, 32]             │         9176576 │
│ ConvTranspose2d                │ [1, 256, 64, 64]             │          524544 │
│ Conv2d                         │ [1, 256, 64, 64]             │         1179904 │
│ ReLU                           │ [1, 256, 64, 64]             │               0 │
│ Conv2d                         │ [1, 256, 64, 64]             │          590080 │
│ ReLU                           │ [1, 256, 64, 64]             │               0 │
│ Sequential                     │ [1, 256, 64, 64]             │         1769984 │
│ DoubleConv                     │ [1, 256, 64, 64]             │         1769984 │
│ DecoderBlock                   │ [1, 256, 64, 64]             │         2294528 │
│ ConvTranspose2d                │ [1, 128, 128, 128]           │          131200 │
│ Conv2d                         │ [1, 128, 128, 128]           │          295040 │
│ ReLU                           │ [1, 128, 128, 128]           │               0 │
│ Conv2d                         │ [1, 128, 128, 128]           │          147584 │
│ ReLU                           │ [1, 128, 128, 128]           │               0 │
│ Sequential                     │ [1, 128, 128, 128]           │          442624 │
│ DoubleConv                     │ [1, 128, 128, 128]           │          442624 │
│ DecoderBlock                   │ [1, 128, 128, 128]           │          573824 │
│ ConvTranspose2d                │ [1, 64, 256, 256]            │           32832 │
│ Conv2d                         │ [1, 64, 256, 256]            │           73792 │
│ ReLU                           │ [1, 64, 256, 256]            │               0 │
│ Conv2d                         │ [1, 64, 256, 256]            │           36928 │
│ ReLU                           │ [1, 64, 256, 256]            │               0 │
│ Sequential                     │ [1, 64, 256, 256]            │          110720 │
│ DoubleConv                     │ [1, 64, 256, 256]            │          110720 │
│ DecoderBlock                   │ [1, 64, 256, 256]            │          143552 │
│ Conv2d                         │ [1, 1, 256, 256]             │              65 │
│ UNet                           │ [1, 1, 256, 256]             │        31031745 │
└────────────────────────────────┴──────────────────────────────┴─────────────────┘
```
    
- **Training** : Uses MFSD dataset, resized to 128x128, trained with Adam optimizer and binary cross-entropy loss for 10 epochs.

#### D.ii: Compare U-Net with Traditional Method

- **Evaluation** : IoU and Dice scores are computed on validation set and compared with traditional segmentation results.

## Hyperparameters and Experiments
-------------------------------
### CNN (Task B)
- We tried a total of 12 different hyperparameters experiments by using different learning rates, optimizer, batch size, dropout rates and final activation functions.
- All the hyperparameters combinations along with their results are stored in **all_cnn_hyperparameters.csv**.
- The best hyperparameters combination along with its result is stored in **best_cnn_hyperparameters.csv**.
- **All hyperparameter configurations used** :
 <img width="989" alt="Screenshot 2025-03-25 at 5 30 12 PM" src="https://github.com/user-attachments/assets/6f0e1880-dc14-4169-a9b9-b13222f88748" />

- **Best hyperparameter configuration** :
 <img width="989" alt="Screenshot 2025-03-25 at 5 31 43 PM" src="https://github.com/user-attachments/assets/c7f084ba-5746-40bf-a3d9-2d9b8288edf4" />


### U-Net (Task D)

- **Experiments Performed**:
  - **Model 1 (Baseline U-Net):**
    - Architecture with fewer filters in encoder and decoder.
    - Learning rate: 0.0005
    - Batch size: 8
    - Loss: Binary Cross-Entropy
    - Epochs: 10
    - **Results:**
        - IoU: 0.78
        - Dice: 0.85
    - **Challenges:**
        - Lower performance due to reduced model capacity.
        - Faster training but led to loss of spatial information.

 - **Model 2 (U-Net with Increased Dropout):**
    - Dropout layers added after each convolutional block.
    - Learning rate: 0.0001
    - Batch size: 16
    - Loss: Binary Cross-Entropy
    - Epochs: 12
    - **Results:**
        - IoU: 0.79
        - Dice: 0.85
    - **Challenges:**
        - Higher dropout led to regularization but caused underfitting.
        - Slightly worse segmentation results due to reduced model capacity.

- **Model 3 (Shallower U-Net):**
    - Encoder and decoder with 3 blocks instead of 4.
    - Learning rate: 0.0005
    - Batch size: 8
    - Loss: Binary Cross-Entropy
    - Epochs: 8
    - **Results:**
        - IoU: 0.81
        - Dice: 0.88
    - **Challenges:**
        - Shallower architecture resulted in loss of fine spatial details.
        - Faster training but underperformed on complex masks.

- **Model 4 (Final U-Net - Best Configuration):**
    - Optimized U-Net with tuned hyperparameters.
    - Learning rate: 0.0001
    - Batch size: 8
    - Loss: Binary Cross-Entropy
    - Epochs: 10
    - **Results:**
        - IoU: 0.84
        - Dice: 0.91
    - **Advantages:**
        - Skip connections improved spatial preservation.
        - Achieved better segmentation performance compared to Model 1.
---

- **Final Hyperparameters** :
 - **Loss Function:** Binary Cross-Entropy
 - **Optimizer:** Adam
 - **Learning Rate:** 0.0001
 - **Batch Size:** 8
 - **Epochs:** 10
 - **Data Split:** 70% Training, 15% Validation, 15% Test
    
- **Experiments** : Single configuration trained due to computational constraints, with early stopping considered but not implemented in the provided code.

---

## Results
-------
### Task A: Traditional ML Classifiers(SVM, Neural Network, XGBoost)

- **SVM** : 93.53% accuracy
- **Neural Network (MLP)** : 91.09% accuracy
- **XGBoost** : 92.43% accuracy

  
  ![Screenshot 2025-03-25 at 5 36 30 PM](https://github.com/user-attachments/assets/11271674-60b6-4348-8849-c77ab6c050d6)


### Task B: Convolutional Neural Network(CNN)

- **CNN** : 95.85% accuracy {'learning_rate': 0.001, 'optimizer': 'adam', 'batch_size': 32, 'final_activation': 'sigmoid', 'dropout_rate': 0.3}
  
  ![Screenshot 2025-03-25 at 5 38 12 PM](https://github.com/user-attachments/assets/c7628180-8aa7-44f8-9404-522a25c48bfc)

### Task C: Traditional Segmentation

#### Avg Accuracy using Region filling algorithm:
- **IoU:** 0.398
- **Dice:** 0.547


  ![Screenshot 2025-03-26 at 12 58 36 AM](https://github.com/user-attachments/assets/f981850d-408f-42aa-88de-93c143f1b92c)


- As the dataset is very large , Otsu thresholding is computationally expensive,  and as its accuracy won't be significantly better than Region-filling algorithm, even though we defined the model, we couldn't run it fully locally on the machine.

#### Sample outputs generated:

- **Region Growing algorithm**:

![image](https://github.com/user-attachments/assets/680bca6d-2797-4a6a-bb88-99f330fd7f77)

- **Otsu Threshold algorithm**: (done on one image, as doing on the entire dataset was very computationally expensive.
- 
<img width="784" alt="Screenshot 2025-03-26 at 1 15 12 AM" src="https://github.com/user-attachments/assets/fa387226-37dd-4bb1-94ef-bdfa7c65d919" />


---

### Task D: U-Net Segmentation


#### Validation Results:
- **Average IoU:** 0.9240
- **Average Dice:** 0.9561

  <img width="372" alt="Screenshot 2025-03-26 at 1 09 23 AM" src="https://github.com/user-attachments/assets/b95bab44-e9fd-4f4a-87b1-e48e76b59bbb" />
#### Sample outputs generated:
- Below are sample visualizations generated from the `test_dataset`:
    - Original Image
    - Ground Truth Mask
    - Model Prediction
    - Overlay (TP/FP/FN) (white/red/blue)

  <img width="400" alt="Result 1" src="https://raw.githubusercontent.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/main/output_viz/result_1127.png" />


  <img width="400" alt="Result 2" src="https://raw.githubusercontent.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/main/output_viz/result_214.png" />


  <img width="400" alt="Result 3" src="https://raw.githubusercontent.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/main/output_viz/result_403.png" />

  
  <img width="400" alt="Result 4" src="https://raw.githubusercontent.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/main/output_viz/result_440.png" />


  <img width="400" alt="Result 5" src="https://raw.githubusercontent.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/main/output_viz/result_777.png" />




---

## Comparison

### Classification
- **CNN Accuracy:** 95.85%  
    - Outperforms SVM (93.53%), MLP (91.09%), and XGBoost (92.43%).  
    - CNN captures spatial features better, giving higher accuracy.

### Segmentation
- **U-Net:** IoU = 0.9240, Dice = 0.9561
    - Extremely high time to run using Otsu, and terrible result using FloodFill which is faster and was used for evaluation
    - Significantly outperforms traditional segmentation methods (IoU = 0.398, Dice = 0.547).
    - Skip connections preserve spatial information, improving segmentation quality.

---

## Observations and Analysis

### Classification
- CNN performs better due to its ability to learn hierarchical spatial features.
- Traditional ML classifiers rely on handcrafted features, which limits their performance.
- **Challenges:**
    - Neural networks required extensive hyperparameter tuning to achieve optimal performance.
    - XGBoost took longer due to the size of the dataset.

### Segmentation
- **U-Net**:
    - Provides precise segmentation with skip connections that retain spatial information.
    - Requires significant computational resources but achieves better segmentation results.
- **Traditional Segmentation**:
    - Otsu thresholding often misidentifies regions with high-intensity variance and is computationally expensive.
    - Region Growing is sensitive to the choice of seed and tolerance, affecting accuracy.
- **Challenges:**
    - U-Net training is computationally intensive.
    - Traditional methods are faster but often misclassify regions.
    - Had to download CUDA and pytorch that is compatible with CUDA to actually be able to train the model, as training on GPU was very difficult

### General
- Deep learning models (CNN, U-Net) outperform traditional approaches but require careful tuning and computational resources.
- Misalignment between image-mask pairs affects segmentation quality.

---

## How to Run the Code

### Prerequisites
- **Python Version:** 3.7+
- **Libraries:** Install via pip:
```
pip install numpy opencv-python matplotlib scikit-learn scikit-image tensorflow torch torchvision
```
- **Datasets:**
    - Download `finaldataset.zip` from GitHub.
    - Download `MSFD.zip` from GitHub.
    - Place both in the same directory as the repository.

---

### Directory Structure
```
VR_Project1_Aryan_Owais_IMT2022102_IMT2022502/
├── BinaryClassification.ipynb
├── Segmentation.ipynb
├── all_cnn_hyperparameters.csv
├── best_cnn_hyperparameters.csv
├── output_viz
│   ├── result_(random image).png
│   └── result_(random image).png
├── README.md
```

---

### Running Classification (BinaryClassification.ipynb)
- Open Notebook:
```
jupyter notebook BinaryClassification.ipynb
```
- Update Paths:
    - Set `zip_file_path` to `finaldataset.zip` and ensure that it's in the same folder as the repository.
- Execute Cells:
    - Run all cells to:
        - Load and preprocess data.
        - Train ML classifiers (Task A).
        - Train and tune CNN (Task B).
- **Outputs:** Accuracy metrics for SVM, MLP, and CNN.

---

### Running Segmentation (Segmentation.ipynb)
- Open Notebook:
```
jupyter notebook Segmentation.ipynb
```
- Update Paths:
    - Set `zip_file_path` to `MSFD.zip` and ensure that it is in the same folder as the repository.
- Execute Cells:
    - Run all cells to:
        - Perform traditional segmentation (Task C).
        - Train and evaluate U-Net (Task D).
- **Outputs:** Visualizations, IoU, and Dice scores.

---


## Notes
- Ensure sufficient RAM and GPU (if available) for U-Net training.
- Outputs are printed in the notebook; no additional intervention is required.
 
