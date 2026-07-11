# X-CapFusion

<img src="./images/logo.png" alt="X-CapFusion Logo" align="right" width="120" />

Lightweight Medical Image Captioning Model - **X-CapFusion** is an advanced **medical image captioning model** designed to generate clinically meaningful and label-aware captions. By integrating visual data, associated medical text, and diagnostic labels, X-CapFusion creates a **label-conditional** captioning system that produces accurate, context-rich descriptions of medical images. Leveraging deep learning and NLP, it supports healthcare professionals in interpreting and analyzing clinical imagery with greater precision and efficiency.

## 🧬 Key Features

- ❌ **NoLabels**: (Figure 1)
  A baseline image captioning model that does **not** incorporate clinical labels at any stage.  
  It relies solely on image features projected into the text decoder feature space to generate captions, without conditioning or label-based loss terms.

  <p align="center">
  <b>Figure 1:</b> No Label Conditioning<br>
  <img src="/images/NoLabels.png" alt="No Label Conditioning" width="200"/>
  </p>

- 🔍 **Intermediate Label Conditioning Image Captioning** (Figure 2):  
  This model uses clinical labels as a *soft prompt* fixed by the phrase "The image shows: (...)".  
  The soft prompt is encoded by a text encoder, while the image is encoded by an image encoder and then projected into the text decoder feature space via a transformer-based mapping network.  
  Both image and label embeddings (projected into the text decoder embedding space) are concatenated through a cross-attention mapper, producing the final text decoder embedding array that feeds into the text decoder to generate the caption.

  <p align="center">
  <b>Figure 2:</b> Intermediate Label Conditioning<br>
  <img src="/images/IntermediateLabel.png" alt="Intermediate Label Conditioning" width="600"/>
  </p>

- 🔍 **Early Label Conditioned Image Captioning** (Figure 3):
  Similar to the intermediate approach, but instead of a fixed soft prompt, it uses *hot labels* that are converted into label embeddings via a Label Mapping Network.  
  These label embeddings are concatenated directly with the image encoding before feeding into the text decoder.

  <p align="center">
  <b>Figure 3:</b> Early Label Conditioning<br>
  <img src="/images/EarlyLabel.png" alt="Early Label Conditioning" width="600"/>
  </p>

- 🔍 **Late Label Conditioned Image Captioning** (Figure 4): 
  This approach applies labels only during the loss calculation phase.  
  The standard image captioning pipeline generates captions from image features projected into the text decoder space.  
  Then, a classifier takes the generated caption as input to predict multilabel outputs.  
  The loss function combines the caption cross-entropy loss with a Binary Cross Entropy loss between the ground truth labels and the predicted labels from the caption.

  <p align="center">
  <b>Figure 4:</b> Late Label Conditioning<br>
  <img src="/images/LateLabel.png" alt="Late Label Conditioning" width="600"/>
  </p>

---


## 📁 Dataset Preparation

Your dataset folder (`MIMIC-CXR/`) should be organized as follows:
```plaintext
MIMIC-CXR/
├── files/
│   └── p17/
│       └── p17393825/
│           └── s50109414/
│               └── ae743223-6e571f2e-fd9c9c2b-37cdbcbe-012af6d0.jpg
├── raw_reports_train.csv
├── raw_reports_test.csv
└── mimic-cxr-2.0.0-negbio.csv
```


### 📄 Report Files

`raw_reports_train.csv` and `raw_reports_test.csv` must have the following structure:

| PATIENT_ID | REPORT_ID       | REPORTS                      |
|------------|-----------------|------------------------------|
| p11924226  | s56353295.txt   | As compared to the .....      |
| p11924226  | s53372149.txt   | Lungs are fully expanded ...  |

### 🏷️ Label File

The `mimic-cxr-2.0.0-negbio.csv` must contain clinical findings in this format:

| subject_id | study_id | Atelectasis | Cardiomegaly | Consolidation | Edema | ... | Support Devices |
|------------|----------|-------------|--------------|---------------|-------|-----|-----------------|
| 10000032   | 50414267 |             |              |               |       |     | 1.0             |
| 10000764   | 57375967 |             |              | 1.0           |       |     | -1.0            |

> ⚠️ If clinical labels are missing, they can be inferred using:
> - **Image Classifier**: Predict labels from images.
> - **Text Classifier**: Predict labels from report text.

---
## 🚀 How to Use

Follow these steps to run the X-CapFusion model and generate medical image captions:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/x-capfusion.git
cd x-capfusion
```

### 2. Prepare the Dataset

Make sure your dataset folder (`MIMIC-CXR/`) is structured as shown above and placed inside the `dataset/` folder of the repo.

```bash
x-capfusion/
├── dataset/
│   └── MIMIC-CXR/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python main.py --model_type $MODEL --mode train
```

Where `$MODEL` is one of:

- `earlyLabel`
- `intermediateLabel`
- `lateLabel`
- `noLabel`
- `imageClassifier`
- `textClassifier`

### 5. Predict Captions or Labels

```bash
python main.py --model_type $MODEL --mode predict --dataset test
```

### 6. Evaluate the Model

```bash
python main.py --model_type $MODEL --mode evaluate --dataset test
```

### 7. View Results

All results will be stored in:
```plaintext
x-capfusion/
├── results/
│   ├── earlyResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── traditional_metrics_test.csv
│   │   └── traditional_metrics_train.csv
│   │   └── medical_metrics_test.csv
│   │   └── medical_metrics_train.csv
│   ├── intermediateResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── traditional_metrics_test.csv
│   │   └── traditional_metrics_train.csv
│   │   └── medical_metrics_test.csv
│   │   └── medical_metrics_train.csv
│   ├── lateResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── traditional_metrics_test.csv
│   │   └── traditional_metrics_train.csv
│   │   └── medical_metrics_test.csv
│   │   └── medical_metrics_train.csv
│   ├── noLabelsResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── traditional_metrics_test.csv
│   │   └── traditional_metrics_train.csv
│   │   └── medical_metrics_test.csv
│   │   └── medical_metrics_train.csv
│   ├── textClassifierResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── metrics_test.csv
│   │   └── metrics_train.csv
│   │   └── dynamic_threshold.json
│   ├── imageClassifierResults/
│   │   └── checkpoint.pt
│   │   └── predictions_train.csv
│   │   └── predictions_test.csv
│   │   └── metrics_test.csv
│   │   └── metrics_train.csv
│   │   └── dynamic_threshold.json
```
---

## 📂 Access Results

You can access the complete experimental results, including model checkpoints, predictions, and evaluation metrics, via the following Google Drive link:

🔗 [Click here to view the results](https://drive.google.com/drive/folders/1ONL2CL720mLRrFMwTXgDERhB0Q6cKMap?usp=sharing)

## 🎥 Demo Video

[![X-CapFusion Demo](/images/DEMO.png)](https://drive.google.com/file/d/1u-GTpDheuy1cB0onu2QTS6KgR9F-zeQv/view?usp=sharing)


## 📌 Citation

If you use this model or dataset setup in your research, please cite:

```bibtex
@misc{amorim2025xcapfusion, title = {X-CapFusion: Lightweight Label-Aware Medical Image Captioning}, author = {Magda T. Amorim and Hugo S. Oliveira and Hélder P. Oliveira and Luís F. Teixeira}, year = {2025}, note = {Available at \url{https://github.com/your-username/x-capfusion}}, }
```
