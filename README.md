# Transformer Emotion Recognition

This repository implements an **emotion recognition** model using **DistilBERT** from Hugging Face's Transformers library. The model is trained on the `emotion` dataset to classify text into multiple emotion categories.

## Repository Link
[Transformer-Emotion-recognition](https://github.com/Osama-Abo-Bakr/Transformer-Emotion-recognition)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The objective of this project is to classify text into different emotion categories (e.g., happiness, sadness, anger, etc.) using a Transformer-based model, specifically DistilBERT. The pipeline includes data preprocessing, tokenization, model training, evaluation, and testing.

## Installation
To use this project, install the required dependencies:

```bash
!pip install transformers torch datasets -U
```

## Dataset
The project uses the **`emotion`** dataset available via the Hugging Face Datasets library. It includes labeled text data for emotion classification.

To load the dataset:
```python
from datasets import load_dataset

# Load emotion dataset
data = load_dataset('emotion')
```

### Dataset Statistics
- Number of labels: `number_of_label`
- Class labels: `class_labels`

## Model Architecture
This project utilizes the **DistilBERT** model for sequence classification. DistilBERT is a smaller and faster variant of BERT, pre-trained on the same corpus but with fewer parameters, making it efficient for fine-tuning.

- **Tokenizer**: `DistilBertTokenizer`
- **Model**: `DistilBertForSequenceClassification`

### Tokenization
The dataset is tokenized using the following configuration:
```python
def token_data(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)
data_encoded = data.map(token_data, batched=True)
```

## Training Process
The model is fine-tuned using the following training arguments:
- **Number of Epochs**: 2
- **Batch Size (Train)**: 8
- **Batch Size (Eval)**: 16
- **Learning Rate Warmup Steps**: 500
- **Weight Decay**: 0.01

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000
)
```

The `Trainer` class is used for training and evaluation:
```python
from transformers import Trainer

def compute_metrics(p):
    pred, label = p
    prediction = pred.argmax(axis=1)
    return {'Accuracy': (prediction == label).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_encoded['train'],
    eval_dataset=data_encoded['validation'],
    compute_metrics=compute_metrics,
)
trainer.train()
```

## Evaluation
The model is evaluated on the test dataset to measure performance:
```python
# Evaluate the model
evaluation = trainer.evaluate(data_encoded['test'])
evaluation

# Generate predictions
from sklearn.metrics import confusion_matrix, accuracy_score

prediction = trainer.predict(data_encoded['test'])[1]
cm = confusion_matrix(data_encoded['test']['label'], prediction)
acc = accuracy_score(data_encoded['test']['label'], prediction)

print('The Accuracy of Testing Data is ---> ', acc*100)
```

## Results
- **Test Accuracy**: `<calculated_accuracy>%`
- **Confusion Matrix**:
```
<confusion_matrix_output>
```

## Usage
To use this project in your applications, clone the repository and follow the steps:
1. Install the dependencies.
2. Load the dataset and preprocess it.
3. Fine-tune the model using the training script.
4. Evaluate the model and use it for predictions.

Example prediction pipeline:
```python
text = "I'm feeling great today!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(axis=1).item()
print("Predicted Emotion:", class_labels[predicted_class])
```

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
