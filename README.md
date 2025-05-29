# Transformers

# Transformer-Based Text Prediction on Harry Potter Book

This project implements a custom Transformer model using TensorFlow and Keras to predict the next word in a sentence based on a sequence of preceding words. The dataset is the first Harry Potter book, and the model is trained to generate text in the style of the book.

---

## 🧠 Project Overview

This repo builds a minimal yet fully functional Transformer model (inspired by the architecture from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper) using TensorFlow 2.x. 
It uses self-attention and positional encoding to understand context and sequence in natural language.

---

## 📚 Dataset

- **Source**: [Harry Potter Books Dataset on Kaggle](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)
- **File Used**: `hp_1.txt` (The Philosopher’s Stone)

The dataset is preprocessed to lowercase and tokenized to build training sequences of length 50, where the model predicts the 51st word.

---

## 🏗️ Architecture Components

### 1. **Tokenization & Input Preparation**
- The full text is tokenized using `Tokenizer` from Keras.
- Input sequences of 50 tokens are created, and the target is the next token.
- Targets are one-hot encoded using `to_categorical`.

### 2. **Embedding Layer**
- `TokenAndPositionEmbedding`: Adds both word token embeddings and positional embeddings to the input sequence.

### 3. **Transformer Block**
- `MultiHeadAttention`: Applies scaled dot-product attention in parallel heads.
- `Feed-Forward Network`: Two dense layers for transforming attention output.
- Residual connections + Layer Normalization.

### 4. **Output Layer**
- Final `Dense` layer with softmax activation to predict one of the vocabulary tokens.

---

## 🧾 Model Summary

- Input: (None, 50) → Embedding → Transformer Block → Dense Softmax (vocab size)
- Total Parameters: ~1.9M
- You can adjust epochs, embed_dim, num_heads, ff_dim, and batch_size to experiment with performance

## 📈 Usage
Once trained, you can use the model to generate the next word given a prompt like:

```
seed_text = "harry looked at the"
sequence = tokenizer.texts_to_sequences([seed_text])[0]
sequence = pad_sequences([sequence], maxlen=50, padding='pre')
predicted = model.predict(sequence)

```
