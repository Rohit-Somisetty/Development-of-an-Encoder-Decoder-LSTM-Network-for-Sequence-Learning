# 🧠 LSTM Arithmetic Calculator ➕➖

Can a neural network learn basic math like a human? This project explores that question using an LSTM (Long Short-Term Memory) network to solve simple addition and subtraction problems presented as text!

## 🚀 Project Goal

The main goal is to build and train a sequence-to-sequence (Seq2Seq) model that takes an arithmetic problem like `"34+56"` as input and predicts the correct answer, also as a sequence of characters, like `"+90 "`. It's a fun way to apply LSTMs to a task beyond typical language processing.

## ✨ What it Does

* **Solves Basic Math:** Handles addition (`+`) and subtraction (`-`) for numbers between 0 and 99.
* **Character-Level Genius:** Processes the math problems one character at a time ('0'-'9', '+', '-', ' ').
* **Encoder-Decoder Power:** Uses a standard LSTM Encoder-Decoder architecture to map the problem sequence to the answer sequence.
* **Reversal Trick:** Investigates whether reversing the input/output sequences (e.g., processing `"65+43"` as `"34+56"`) helps the LSTM learn faster or better – a neat trick often used in Seq2Seq tasks!
* **High Accuracy:** Achieves impressive accuracy (around 96-97% on test data) in learning these arithmetic rules from scratch.

## 🧠 The Core Idea: Sequence-to-Sequence Learning

Think of this like translation: the model "translates" the sequence of characters representing the *problem* into a sequence of characters representing the *answer*.

1.  **Encoder:** An LSTM reads the input sequence (e.g., `['5', '-', '2', '0', ' ']`) and compresses its meaning into a fixed-size "thought vector" (its final hidden state).
2.  **Decoder:** Another LSTM takes this thought vector and generates the output sequence step-by-step (e.g., `['-', '1', '5', ' ']`).

## 🔄 Standard vs. Reversed Input

We trained two models:
1.  **Baseline:** Input: `"12+5 "` -> Target: `"+17 "`
2.  **Reversed:** Input: `" 5+21"` -> Target: `" 71+"` (Both input and target are reversed!)

Why reverse? Sometimes, reversing the input creates shorter dependencies between related input/output characters (like the last digit of the input and the first digit of the output), making it easier for the LSTM to learn. Our results show this might offer a slight advantage!

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib

*(You could add badges here if you like, e.g., using shields.io)*
*Example:*
`![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)`
`![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)`

## 📊 Results & Insights

Both models learned the arithmetic tasks successfully, reaching **~96-97% accuracy** on unseen test problems! The comparison plot (see below) suggests that the model trained on reversed sequences might converge slightly faster or achieve marginally better validation accuracy.

## 📈 Visuals

[Model Accuracy Comparison](accuracy_plot.png)
