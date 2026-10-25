# HR.DAYA: Facial Emotion Classification API

**Senior Capstone (2024-2025), NASA HUNCH-related project (Voyager Wellness)**

Using AI to track the psychological effects of isolation and stressful missions on astronauts by classifying emotions from onboard imagery. This repo contains the model-training notebooks, the project posters, and a small **FastAPI service** that serves the trained model.

- [Poster 1 (PDF)](Poster1.pdf)
- [Poster 2 (PDF)](Poster2.pdf)
- [Trained model download (Releases)](https://github.com/nimasriram/neuro_nova/releases/latest)

## Research question

How can AI track the psychological impacts of isolation, stressful missions, and the physical effects of microgravity by assessing astronauts' emotions using onboard imagery?

---

## Run the API

The service takes a face image and returns the predicted emotion with a probability for each class.

### Requirements
- Python 3.12 (developed and tested on Python 3.12, Windows)
- The trained model file `hr_daya.keras`

### 1. Get the code
```bash
git clone https://github.com/nimasriram/neuro_nova.git
cd neuro_nova
```

### 2. Add the model
Download `hr_daya.keras` from the [latest release](https://github.com/nimasriram/neuro_nova/releases/latest) and place it here:
```
app/model/hr_daya.keras
```

### 3. Create a virtual environment and install dependencies

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` and try again.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install includes TensorFlow, so it can take a few minutes.

### 4. Start the server
Run this from the repo root (the `neuro_nova` folder):
```bash
uvicorn app.main:app --reload
```
The first start takes a moment while TensorFlow loads. The server is ready when you see `Application startup complete`.

### 5. Try it
Open the interactive docs at **http://127.0.0.1:8000/docs**, choose **POST /predict**, click **Try it out**, upload a face image (`.jpg` or `.png`, up to 5 MB), and click **Execute**.

Or use the command line:

```bash
# macOS / Linux
curl -X POST -F "file=@face.jpg" http://127.0.0.1:8000/predict
```
```powershell
# Windows PowerShell (use curl.exe, not curl)
curl.exe -X POST -F "file=@face.jpg" http://127.0.0.1:8000/predict
```

Example response:
```json
{
  "label": "N",
  "emotion": "Neutral",
  "confidence": 0.82,
  "probabilities": {"A": 0.05, "F": 0.04, "H": 0.09, "N": 0.82}
}
```

Stop the server with `Ctrl + C`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` when the service is running |
| `POST` | `/predict` | Upload an image (form field `file`) and get the predicted emotion |
| `GET` | `/docs` | Interactive API documentation |

**Errors:** `400` if the file is not a valid image, `413` if the image is larger than 5 MB.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Could not import module "app.main"` | Run the command from the repo root, not from inside `app/` |
| `No such file ... hr_daya.keras` | Place the model at `app/model/hr_daya.keras` |
| Model fails to load | Install the exact versions in `requirements.txt`; saved Keras models can fail to load across versions |
| Install error for TensorFlow | Check your Python version with `python --version`; use 3.12 |

---

## Project structure
```
neuro_nova/
  app/
    main.py              FastAPI service
    model/hr_daya.keras  trained model (download from Releases)
  PS_Model.ipynb         data preparation and model training
  PS_streamlit_app.ipynb Streamlit demo app (with feedback loop)
  Poster1.pdf, Poster2.pdf
  requirements.txt
```

## Model

- **Task:** facial emotion recognition (image classification) with a Convolutional Neural Network called HR.DAYA
- **Classes (output order):** `A` angry, `F` fearful, `H` happy, `N` neutral
- **Input:** RGB image resized to 300 x 200 pixels (array shape `(200, 300, 3)`), pixel values scaled to 0-1
- **Architecture:** three convolution + max-pooling blocks (32, 64, 128 filters), a 64-unit dense layer, and a 4-way softmax
- **Dataset:** Chicago Face Database (CFD), high-resolution, diverse faces
- **Data preprocessing:** label extraction from filenames, one-hot encoding, checking class skew (e.g., 597 of 1,207 CFD images were "neutral"), data augmentation with Gaussian and unsharp filters
- **Tools:** Python, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, FastAPI, Google Colab

## Results

- The model was trained 4 times (10 epochs twice, 15 epochs twice).
- Accuracy ranged from **54-60%** (first runs) to **74-79%** (later runs), with a best of **83% after 15 epochs**.
- Engineering goal: 60-80% accuracy plus an app for testing. Both were built.
- A **Streamlit app** lets users capture an image, review the predicted emotion, and mark it correct or incorrect. Incorrect predictions are added to a dataset used for retraining.

Retraining with the notebooks may give slightly different numbers.

## Limitations and ethics

- The dataset had many more neutral images than other emotions (label skew).
- Training and validation loss was high, so the model may perform well on its own training data but not on real-world scenarios.
- All training images are posed, so expressions may be exaggerated, which likely led to some overfitting.
- Reading emotion from a face is an imperfect and sensitive task. This is a research prototype and **should not be used for medical, mental-health, or hiring decisions.**

## Training notebooks

The notebooks were written in Google Colab and mount Google Drive. To run them, download the dataset below, update the file paths near the top of each notebook, and install the libraries listed above.

## Data and licensing

Datasets are not included in this repository. Please obtain them from their original sources under their own terms (Chicago Face Database, which is for research use).

## Future research

- I looked into using pupil dilation to estimate mental state. It did not produce usable results, and I am concerned it may be unreliable.
- I tried a dataset with multiple frames of a person changing expression. This may suit the project's goals better in future work.
- Recording a dataset with wearable cameras such as smart glasses could capture natural expressions, since a main limitation was that all dataset images were staged. This would require hours of footage and raises ethical concerns, so it would need the informed consent of everyone recorded.
- Overall, there are ethical concerns about using AI to read a person's facial expression.

## Author

Purnima Sriram, University of Texas at Dallas