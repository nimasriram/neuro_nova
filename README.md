# neuro_nova

**Senior Capstone (2024-2025), NASA HUNCH-related project**

Using AI to track the psychological effects of isolation and stressful missions on astronauts by classifying emotions from onboard imagery.

## Research question

How can AI track the psychological impacts of isolation, stressful missions, and the physical effects of microgravity by assessing astronauts' emotions using onboard imagery?

## Approach

**Task:**
facial emotion recognition (image classification) by building a Convolutional Neural Network (CNN) called HR.DAYA

**Datasets evaluated:**
Chicago Face Database (CFD), high-resolution, diverse faces

**Data preprocessing:**
label extraction from filenames, one-hot encoding, checking class skew (e.g., 597 of 1,207 CFD images were "neutral"), data augmentation with Gaussian and unsharp filters

**Tools:**
Python, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, Google Colab

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

## Future research

During this project, I did look into the potential to use pupil dialtion to determine the state of someone's mental health. Although it didn't result in anything, I am concerned that this may be somewhat unreliable. 

Additionally, I attempted to utilize a dataset that had multiple frames of a person changing their expression. This may be more helpful for the scope of this project for future research. 

Finally, with the advent of Meta glasses that have the ability to record individuals, creating a dataset through this device may be the most useful for this project. However this would require hours of recordings, and there are ethical concerns regarding the use of Meta glasses. If a person who is being recorded consents to being recorded, it may be useful as it may capture their natural expressions. One of the main limitations through the datasets was that all the images were staged and the expressions may have been exaggerated to some degree which has lead to some overfitting in the model. 

Overall there are ethical concerns regarding the use of AI reading a human's facial expression. 


## Author

Purnima Sriram