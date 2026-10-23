# neuro_nova

**Senior Capstone (2024-2025), NASA HUNCH-related project**

Using AI to track the psychological effects of isolation and stressful missions on astronauts by classifying emotions from onboard imagery.

## Research question

How can AI track the psychological impacts of isolation, stressful missions, and the physical effects of microgravity by assessing astronauts' emotions using onboard imagery?

## Approach

- **Task:** facial emotion recognition (image classification) with a Convolutional Neural Network (CNN)
- **Datasets evaluated:**
  - Face Expression Recognition Dataset (Kaggle, 35k+ images, 48x48, 7 emotions)
  - Chicago Face Database (CFD), high-resolution, diverse faces
- **Data preprocessing:** label extraction from filenames, one-hot encoding, checking class skew (e.g., 597 of 1,207 CFD images were "neutral"), data augmentation with Gaussian and unsharp filters
- **Tools:** Python, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, Google Colab

## Repository contents
| Path | What it is |
|---|---|
| `docs/Neuro_Novasearch_Paper.docx` | Full research paper: background, literature review, methodology |
| `PS_Model.ipynb`| Final model notebook (including audio exploration with librosa)

## Results

> TODO: add your final accuracy / validation results, one confusion matrix or training curve image, and 2-3 sentences on what you learned.

## Limitations and next steps
> TODO: e.g., dataset skew, small CFD sample, lab-posed vs. real astronaut conditions, privacy considerations.

## Running the notebooks
The notebooks were written in Google Colab and mount Google Drive. To run them, download the datasets above, update the file paths near the top of each notebook, and install the libraries listed above.

## Data and licensing
Datasets are not included in this repository. Please obtain them from their original sources under their own terms (Kaggle; Chicago Face Database, which is for research use).

## Author
Purnima Sriram, University of Texas at Dallas
