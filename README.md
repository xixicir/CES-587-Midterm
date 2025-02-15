# CES 587 Midterm Project - Spam Detection Using CNN and GloVe Embeddings

## 1. Overview
This project uses a Convolutional Neural Network with GloVe word embeddings to classify SMS messages as spam or not spam. The model is trained on the SMS Spam Collection Dataset and implemented in Python using TensorFlow/Keras.

## 2. Getting Started
To reproduce the results, follow the steps below.

### (1) Clone the GitHub Repository
First, download the project from GitHub:

```bash
git clone git@github.com:xixicir/CES-587-Midterm.git
```

### (2) Run on Google Colab
This project is designed to run on Google Colab for easy execution without local setup:
1. Upload the `CES 587 Midterm.ipynb` notebook to Google Colab.

2. Run each cell step by step:

3. Install dependencies, download dataset, train and evaluate the model
Run the first cell in Colab to install dependencies, download the dataset, train and evaluate the model. This will require you to upload your kaggle.json file for authentication:

```python
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Install Kaggle
!pip install kaggle
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d uciml/sms-spam-collection-dataset
!unzip sms-spam-collection-dataset.zip

import os
if not os.path.exists("spam.csv"):
    !mv sms-spam-collection-dataset/spam.csv .

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

if not os.path.exists("glove.6B.100d.txt"):
    !wget http://nlp.stanford.edu/data/glove.6B.zip
    !unzip glove.6B.zip

glove_file = "glove.6B.100d.txt"
embeddings_index = {}
with open(glove_file, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential([
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = padded_sequences
y_train = df['label'].values

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, X_train, y_train)
```

4. Run the second cell to test the model with new SMS messages:

```python
new_texts = ["Congratulations! You won a free iPhone. Click here!", "Hey, are we meeting tomorrow?"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
predictions = model.predict(new_padded)
print(predictions)
```