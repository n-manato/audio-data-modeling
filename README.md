# Speech Emotion Recognition from Audio

## Overview

This project aims to classify human emotions from speech audio.  
Using speech data from the RAVDESS dataset, I extracted several types of audio features and compared multiple machine learning models to examine which combinations work effectively for speech emotion recognition.

The target emotions in this project are:

- neutral
- happy
- sad
- angry

This project demonstrates the overall workflow of an audio machine learning task, including:

- audio loading
- visualization
- feature extraction
- model training
- evaluation

---

## Processing Flow

The overall process of this project is shown below.

```mermaid
flowchart TD
    A[Load audio files<br>from RAVDESS dataset] --> B[Extract emotion labels<br>from file names]
    B --> C[Visualize audio data<br>Waveform / Spectrogram / MFCC / etc.]
    C --> D[Extract features<br>Pitch / Spectrogram / Mel Spectrogram / MFCC]
    D --> E[Create dataset<br>X: features, y: labels]
    E --> F[Split into training and test sets]
    F --> G[Train models<br>Random Forest / SVM / Logistic Regression / LSTM]
    G --> H[Evaluate models<br>Accuracy / Classification Report / Confusion Matrix]
    H --> I[Compare results<br>and analyze performance]

## Project Theme Selection

Before deciding on the final topic, I considered several possible ideas based on hearing, sound, and daily-life applications:

- analysis using car engine sounds
- classification of car horn sounds
- classification of car brands or manufacturers from engine sound
- classification of engine RPM (1000 / 1500 / 2000 rpm) from sound
- estimation of engine RPM from sound
- engine sound based condition monitoring or anomaly detection
- emotion recognition from speaking style
- occupation classification from speaking style
- pet health monitoring from breathing sound
- temperature estimation from shower sound
- wind speed estimation from wave sound

Among these ideas, I chose **speech emotion recognition**.

### Why I chose this theme

I chose emotion recognition because I became interested in whether audio analysis could help understand a speaker’s internal state from the way they speak.  
Recently, I have had more opportunities to talk with people, and sometimes I feel that it is difficult to understand what the other person is thinking only from the words themselves.  
This made me think that emotional cues in speech, such as tone, pitch, and energy, may contain useful information.  

From a technical perspective, this theme is also suitable because emotion is reflected in multiple acoustic properties, which makes it a good problem for comparing different features and machine learning methods.

---

## Objective

The objective of this project is to classify emotions from speech audio by:

1. loading and exploring audio files
2. visualizing audio signals
3. extracting multiple types of audio features
4. training and comparing machine learning models
5. evaluating the models using appropriate classification metrics

The target emotions used in this project are:

- neutral
- happy
- sad
- angry

---

## Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

In the experiments, I selected speech files corresponding to four emotions:

- `01` = neutral
- `03` = happy
- `04` = sad
- `05` = angry

These classes were chosen because they provide clear emotional differences while keeping the classification problem manageable for comparison across multiple feature extraction methods.

---

## Overall Program Flow

The overall workflow of the project is as follows:

1. **Load the dataset**  
   Read `.wav` files from the RAVDESS dataset directory.

2. **Parse labels from file names**  
   Extract emotion codes from file names and map them to emotion labels.

3. **Explore and visualize audio**  
   Plot waveforms and other visual/audio features to understand the characteristics of the data.

4. **Extract features**  
   Convert variable-length audio signals into machine-readable numerical features.

5. **Prepare the dataset**  
   Build feature matrix `X` and label vector `y`.

6. **Split the data**  
   Divide the dataset into training and test sets using stratified splitting.

7. **Train models**  
   Train several machine learning models on the extracted features.

8. **Evaluate the models**  
   Compare models using classification metrics such as accuracy, classification report, and confusion matrix.

9. **Compare feature-model combinations**  
   Observe which features and models work better for speech emotion recognition.

---

## Audio Visualization / Feature Candidates Considered

To understand the data and to examine possible feature extraction methods, I reviewed the following audio representations and signal-based descriptors.

### Visualization and feature candidates

- Waveform
- Spectrogram
- RMS Energy
- Zero Crossing Rate
- Pitch Contour
- Mel Spectrogram
- MFCC
- Delta MFCC
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Contrast
- Chroma
- Harmonic and Percussive Components

These were useful because they represent different aspects of audio:

- **time-domain behavior**: waveform, RMS energy, zero crossing rate
- **frequency-domain behavior**: spectrogram, spectral centroid, bandwidth, rolloff, contrast
- **perceptual speech features**: mel spectrogram, MFCC, delta MFCC
- **pitch and tonal information**: pitch contour, chroma
- **signal decomposition**: harmonic and percussive components

---

## Feature Extraction Methods Considered

In this project, I implemented and compared several feature extraction approaches.

### 1. Pitch Contour Based Features
This approach uses the pitch trajectory of speech and summarizes it with statistical values such as mean and standard deviation.

**Why it is useful:**  
Pitch is closely related to prosody and intonation, which often change with emotion. For example, anger and happiness may involve stronger pitch variation than neutral speech.

**Limitation:**  
Pitch alone may not capture enough information about timbre, resonance, and spectral shape. Also, pitch can be unstable in unvoiced regions.

---

### 2. Spectrogram Based Features
This approach uses the time-frequency representation from STFT and summarizes it statistically.

**Why it is useful:**  
A spectrogram captures how frequency energy changes over time, which is important because emotional speech often changes not only in pitch but also in energy distribution across frequencies.

**Limitation:**  
A raw spectrogram can contain a large amount of information, some of which may be redundant or sensitive to noise.

---

### 3. Mel Spectrogram Based Features
This approach converts the spectrogram into the mel scale and summarizes it using statistics such as mean and standard deviation for each mel band.

**Why it is useful:**  
The mel scale is closer to human auditory perception, so it can represent speech characteristics in a more perceptually meaningful way than a raw spectrogram.

**Limitation:**  
Although it is more compact than a standard spectrogram, it still may not capture speech structure as efficiently as MFCC for some classification tasks.

---

### 4. MFCC Based Features
This approach extracts MFCCs and summarizes them using statistical descriptors such as mean and standard deviation.

**Why it is useful:**  
MFCC is one of the most widely used features in speech analysis because it captures the spectral envelope of speech in a compact form.  
It is especially suitable for tasks involving speech characteristics, speaker traits, and emotional expression.

**Strength:**  
MFCC provides a good balance between compactness and expressiveness, making it appropriate for classical machine learning models.

---

### 5. MFCC Sequence for LSTM
Instead of reducing MFCCs into summary statistics, this approach keeps the MFCC sequence over time and feeds it into an LSTM model.

**Why it is useful:**  
Emotion is not only reflected in average acoustic properties but also in how speech evolves over time.  
An LSTM can model temporal patterns and sequence dynamics, which may contain useful emotional information.

**Limitation:**  
Deep learning models generally require more computation and may need more data or tuning than classical machine learning methods.

---

## Features Selected for This Project

The implemented feature extraction methods in this project are:

- Pitch contour features
- Spectrogram-based features
- Mel spectrogram-based features
- MFCC-based features
- MFCC sequence features for LSTM

### Why these features were selected

These features were selected because together they cover multiple important viewpoints of speech audio:

- **Pitch contour** captures prosody and intonation
- **Spectrogram** captures general time-frequency patterns
- **Mel spectrogram** captures perceptually meaningful spectral energy
- **MFCC** captures compact speech-specific spectral features
- **MFCC sequences for LSTM** preserve temporal structure

This combination makes the comparison more meaningful because it includes both:

- simple statistical features
- richer time-frequency features
- classical machine learning input
- sequence-based deep learning input

In other words, the selected features were chosen to compare **different levels of abstraction** in audio representation rather than relying on only one kind of descriptor.

---

## Why I Did Not Use All Candidate Features as Main Inputs

Although many other audio descriptors were explored, I did not use all of them as the main classification features.

For example:

- **Waveform** is useful for visualization, but raw waveform values are not always the most compact or stable input for simple classifiers.
- **RMS Energy** is informative but too limited by itself for multi-class emotion recognition.
- **Zero Crossing Rate** can reflect signal noisiness or frequency tendency, but it is usually not sufficient alone for emotion classification.
- **Spectral centroid / bandwidth / rolloff / contrast** are useful supporting descriptors, but individually they may not represent emotional speech as comprehensively as MFCC or mel-based features.
- **Chroma** is often more useful for music-related tonal analysis than for spoken emotion classification.
- **Harmonic/percussive decomposition** is interesting for analysis, but it is less directly suited as a primary compact feature set in this project.

Therefore, I treated these as important exploratory and analytical tools, while selecting features that are more standard and practical for speech emotion classification experiments.

---

## Machine Learning Algorithms Considered

For the classification stage, I considered both classical machine learning models and a deep learning model.

### Classical machine learning models
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

### Deep learning model
- LSTM

---

## Machine Learning Models Selected and Why

### 1. Random Forest
Random Forest was selected because it is a strong baseline model for tabular features.

**Why I used it:**
- it handles nonlinear relationships
- it is relatively robust to noise
- it works well even without heavy parameter tuning
- it can perform reasonably well on summarized statistical features

**Why it makes sense here:**  
Many extracted features are aggregated into fixed-length vectors, and Random Forest is well suited for this type of structured input.

---

### 2. Support Vector Machine (SVM)
SVM was selected because it is a standard and effective model for medium-sized classification problems.

**Why I used it:**
- it often performs well on high-dimensional feature spaces
- it can model nonlinear class boundaries with kernels such as RBF
- it is commonly used in speech and audio classification tasks

**Why it makes sense here:**  
Features such as MFCC, mel spectrogram summaries, and spectrogram summaries may form complex boundaries between emotions, and SVM is suitable for that kind of separation.

---

### 3. Logistic Regression
Logistic Regression was selected as a simple linear baseline.

**Why I used it:**
- it is easy to interpret
- it provides a useful baseline for comparison
- it helps show whether the problem can be solved with a simpler linear model

**Why it makes sense here:**  
Including Logistic Regression makes the comparison more convincing, because it shows whether more complex nonlinear models are actually necessary.

---

### 4. LSTM
LSTM was selected for sequence-based modeling using MFCC time series.

**Why I used it:**
- it can model temporal dependencies
- emotion may appear in the way speech changes over time, not only in averaged statistics
- it provides a comparison between classical feature-based learning and sequence-based deep learning

**Why it makes sense here:**  
Speech is inherently sequential, so it is reasonable to test whether preserving time order improves emotion classification.

---

## Why I Used These Models Together

These models were selected as a set because they represent different modeling assumptions:

- **Logistic Regression**: linear baseline
- **Random Forest**: nonlinear ensemble model for tabular data
- **SVM**: strong nonlinear classifier for feature vectors
- **LSTM**: sequence model for temporal speech patterns

By comparing them, I can show that I did not choose one model arbitrarily.  
Instead, I tested models with different strengths to examine which type of learning is more suitable for this task.

---

## Evaluation Methods Considered

Because this is a **multi-class classification** problem, the following evaluation methods are important:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### Why these metrics are necessary

- **Accuracy** gives an overall measure of correct predictions.
- **Precision** shows how reliable the positive predictions are for each class.
- **Recall** shows how well the model captures actual examples of each class.
- **F1-score** balances precision and recall.
- **Classification Report** summarizes precision, recall, and F1-score for all classes.
- **Confusion Matrix** shows which emotions are confused with each other.

---

## Evaluation Methods Selected in This Project

In the current implementation, I mainly used:

- Accuracy
- Classification Report
- Confusion Matrix

### Why these were selected

These metrics were selected because together they provide both:

- a simple overall performance measure
- a detailed class-by-class error analysis

This is important in emotion recognition, because a model may have reasonable overall accuracy while still confusing certain emotions, such as:

- happy vs. angry
- sad vs. neutral

The confusion matrix is especially useful because it reveals the actual error pattern instead of showing only a single summary score.

---

## Implementation Summary

### Implemented feature extraction pipelines
- Pitch contour based statistical features
- Spectrogram based statistical features
- Mel spectrogram based statistical features
- MFCC based statistical features
- MFCC sequence features for LSTM

### Implemented models
- Random Forest
- SVM
- Logistic Regression
- LSTM

### Implemented evaluations
- Accuracy
- Classification Report
- Confusion Matrix

---

## Files in This Project

### Main experiment notebooks
- `main_pitchcountour.ipynb`
- `main_spectrogram.ipynb`
- `main_melspectrogram.ipynb`
- `main_mfcc.ipynb`
- `main_mfcc_lstm.ipynb`

### Visualization notebook
- `audio_visualization.ipynb`

The visualization notebook was used to explore and summarize candidate audio representations, while the main notebooks were used to run classification experiments with different feature extraction methods.

---

## What This Project Demonstrates

This project demonstrates that I understand the following points in audio machine learning:

- how to load and organize speech audio data
- how to visualize audio signals in multiple ways
- how to compare different audio feature extraction methods
- how to choose features according to the task
- how to compare multiple machine learning algorithms
- how to evaluate a classification model appropriately
- how to build a full workflow from raw audio to prediction and evaluation

In other words, this project is not only about obtaining a result, but also about showing a clear and logical process of problem setting, feature selection, model selection, and evaluation.

---

## Future Work

Possible future improvements include:

- using more emotions from the full RAVDESS dataset
- combining multiple feature types into one feature vector
- tuning model hyperparameters
- applying cross-validation
- testing data augmentation
- comparing CNN- or Transformer-based audio models
- analyzing which emotions are most difficult to distinguish and why

---

## Conclusion

In this project, I chose **speech emotion recognition** as the final theme and approached it as a comparison problem across multiple feature extraction methods and machine learning models.

Rather than selecting one method immediately, I examined a broad set of audio representations, selected several meaningful feature extraction pipelines, compared multiple classification models, and evaluated them using standard multi-class metrics.

This project allowed me to demonstrate knowledge of both:

- **audio signal analysis**
- **machine learning model design and evaluation**

and to organize them into a complete, logical workflow.