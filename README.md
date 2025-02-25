<<<<<<< HEAD
# IntentDetection
=======
# Intent Detection

This project implements an intent detection model using BERT (Bidirectional Encoder Representations from Transformers). The model is trained to classify user queries into predefined intent categories, making it suitable for conversational AI applications.

## Features
- Uses `bert-base-uncased` for text representation.
- Implements a transformer-based classifier for intent detection.
- Preprocesses text data by tokenizing and normalizing.
- Provides a trained model for real-time inference.

## Dataset
The dataset consists of user queries mapped to 21 intent categories. It includes 328 labeled entries. The text data is preprocessed and split into training (80%) and validation (20%) sets.

## Installation
Ensure you have Python 3.7+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Training the Model
Run the following command to train the model:

```bash
python train.py
```

This will:
- Load and preprocess the dataset.
- Tokenize inputs using BERT tokenizer.
- Train a transformer-based classifier.
- Save the trained model and label encoder.

## Inference
To test the model on new user inputs, run:

```bash
python inference.py --text "What is the weather like today?"
```

This will output the predicted intent category for the given input text.

## Model Architecture
- Pre-trained BERT (`bert-base-uncased`) as the feature extractor.
- A dropout layer to prevent overfitting.
- A fully connected layer for classification.

## Hyperparameters
- Optimizer: AdamW (`lr=2e-5`)
- Batch size: 8
- Epochs: 5

## Improvements & Future Work
- Hyperparameter tuning for better performance.
- Expanding the dataset with more training samples.
- Exploring ensemble models (e.g., combining BERT with LSTMs).
- Domain-specific fine-tuning for better generalization.

## License
This project is open-source and available under the MIT License.

>>>>>>> 1a75681 (Committing)
