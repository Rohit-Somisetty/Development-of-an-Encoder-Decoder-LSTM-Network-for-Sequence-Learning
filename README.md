# LSTM Arithmetic Calculator

## Description

This project implements an Encoder-Decoder LSTM (Long Short-Term Memory) network using Keras/TensorFlow to solve simple arithmetic problems presented as character sequences. The model learns to perform addition and subtraction on two-digit numbers (e.g., input "34+56" should predict " +90").

The core idea is to treat the arithmetic problem as a sequence-to-sequence task, where the input sequence is the problem string (e.g., "a op b") and the output sequence is the answer string[cite: 2, 7, 8, 9]. The project also explores the effect of reversing the input and output sequences during training, a technique known to sometimes improve performance on sequence tasks[cite: 2, 15].

## Key Features

* Solves simple addition (+) and subtraction (-) problems for numbers between 0 and 99[cite: 2].
* Employs an Encoder-Decoder LSTM architecture for sequence-to-sequence learning[cite: 7, 11, 18].
* Processes problems at the character level, encoding each character ('0'-'9', '+', '-', ' ') into a vector[cite: 4].
* Compares the performance of the model trained on standard sequences versus reversed sequences[cite: 20].
* Achieves high accuracy (~96-97%) on the test set for both variations[cite: 13].

## Technologies Used

* Python
* Keras / TensorFlow [cite: 1]
* NumPy [cite: 1]
* Scikit-learn (for data splitting) [cite: 1]
* Matplotlib (for plotting results) [cite: 1]

## Dataset

The dataset is synthetically generated. It consists of pairs of arithmetic problems (like "i+j" or "i-j" where i and j are numbers from 0 to 99) and their corresponding answers[cite: 2].
* Input strings (queries) are padded to a fixed length of 5 characters (e.g., "5-20 ").
* Output strings (answers) are padded/formatted to a fixed length of 4 characters (e.g., "-15 ").
* Each character is converted into a one-hot encoded vector based on the alphabet ['0'...'9', '+', '-', ' '][cite: 4].

## Model Architecture

The model uses an Encoder-Decoder architecture[cite: 7]:
1.  **Encoder LSTM:** An LSTM layer reads the input sequence (length 5, character embedding dimension 13) and outputs a fixed-size context vector (hidden state)[cite: 8, 11].
2.  **RepeatVector:** This layer repeats the context vector from the encoder 4 times (matching the desired output sequence length)[cite: 9, 11].
3.  **Decoder LSTM:** Another LSTM layer takes the repeated context vector sequence as input and generates the output sequence step-by-step[cite: 9, 11].
4.  **Dense Layer:** A TimeDistributed Dense layer with softmax activation is applied to each step of the decoder's output to predict the probability distribution over the character vocabulary for each position in the output sequence[cite: 10, 11].

The loss function used is categorical cross-entropy, optimized with Adam[cite: 11].

## How it Works

1.  **Data Generation:** Pairs of queries (e.g., "12+5") and answers (e.g., "+17") are generated. A separate reversed dataset is also created (e.g., "5+21" -> "71+")[cite: 2, 15].
2.  **Encoding:** Both queries and answers are converted into sequences of one-hot vectors[cite: 4, 5].
3.  **Training:** The Encoder-Decoder LSTM model is trained on the encoded data pairs to minimize the prediction error[cite: 12, 18]. Training is performed separately for the standard and reversed datasets.
4.  **Evaluation:** The models are evaluated on unseen test data to measure their accuracy[cite: 13].
5.  **Comparison:** The validation accuracy curves for both standard and reversed input training runs are plotted to compare their learning dynamics[cite: 20].

## Results

Both the baseline model and the model trained on reversed sequences achieve high test accuracy, around 96-97%[cite: 13]. The visualization comparing validation accuracies suggests that reversing the sequences might lead to slightly faster convergence or better final performance in this specific task[cite: 20].

## How to Run

1.  **Prerequisites:** Ensure you have Python and the necessary libraries installed (TensorFlow/Keras, NumPy, Scikit-learn, Matplotlib).
    ```bash
    pip install tensorflow numpy scikit-learn matplotlib
    ```
2.  **Execute Notebook:** Run the Jupyter Notebook `Assignment_9 (2).ipynb` (based on the provided PDF name) cell by cell to generate data, build the models, train them, and visualize the results.
