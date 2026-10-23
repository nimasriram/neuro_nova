# neuro_nova

**Senior Capstone (2024-2025), NASA HUNCH-related project**

Using AI to track the psychological effects of isolation and stressful missions on astronauts by classifying emotions from onboard imagery.

## Research question

How can AI track the psychological impacts of isolation, stressful missions, and the physical effects of microgravity by assessing astronauts' emotions using onboard imagery?

## Approach

**Task:**
facial emotion recognition (image classification) with a Convolutional Neural Network (CNN)

**Datasets evaluated:**
Chicago Face Database (CFD), high-resolution, diverse faces

**Data preprocessing:**
label extraction from filenames, one-hot encoding, checking class skew (e.g., 597 of 1,207 CFD images were "neutral"), data augmentation with Gaussian and unsharp filters

**Tools:**
Python, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, Google Colab

## Repository contents
| Path | What it is |
|---|---|
| `docs/Neuro_Nova_Research_Paper.docx` | Full research paper: background, literature review, methodology |
| `PS_Model.ipynb`| Final model notebook (including audio exploration with librosa)

## Results

Final accuracy / validation results

- The model was trained 4 times (10 epochs twice, 15 epochs twice).
- Accuracy ranged from **54-60%** (first runs) to **74-79%** (later runs), with a best of **83% after 15 epochs**.
- Engineering goal: 60-80% accuracy plus a Streamlit app for testing. Both were built.
- A **Streamlit app** lets users capture an image, review the predicted emotion, and mark it correct or incorrect. Incorrect predictions are added to a dataset used for retraining.

## Limitations and next steps
- The dataset had many more neutral images than other emotions (label skew).
- Training and validation loss was high, so the model may perform well on its own training data but not on real-world scenarios.
- Next steps: improve accuracy and reduce training and validation loss.

## Running the notebooks
The notebooks were written in Google Colab and mount Google Drive. To run them, download the datasets above, update the file paths near the top of each notebook, and install the libraries listed above.

## Data and licensing
Datasets are not included in this repository. Please obtain them from their original sources under their own terms (Kaggle; Chicago Face Database, which is for research use).


## Author
Purnima Sriram