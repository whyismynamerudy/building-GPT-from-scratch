# Building-GPT-from-scratch: Decoder-Only Transformer for Text Generation

This project implements a decoder-only transformer model using PyTorch, designed for text generation tasks. The primary goal is to generate Shakespeare-like text sequences by predicting the next character in a sequence based on the context of previous characters.

Google Colab Link: [https://colab.research.google.com/drive/1Jwn6bEMDcNY20YBtTdfDgECoHDhqgwko?usp=sharing][https://colab.research.google.com/drive/1Jwn6bEMDcNY20YBtTdfDgECoHDhqgwko?usp=sharing]

## Overview

The implementation includes:

- **Data Preparation:** Loading and preprocessing text data from a source file.
- **Model Architecture:** A decoder-only transformer composed of:
  - **FeedForward:** A simple linear layer followed by non-linearity.
  - **MultiHeadAttention:** Multiple heads of self-attention in parallel.
  - **Block:** Transformer block comprising communication and computation stages.
  - **GPT (Generative Pre-trained Transformer):** Utilizes token and position embeddings along with multiple transformer blocks for prediction.
- **Training:** AdamW optimizer with cross-entropy loss function for training the model.
- **Evaluation:** Periodic evaluation of the model on training and validation datasets.

## Usage

1. **Dependencies:** Ensure you have PyTorch installed.
2. **Data Preparation:** Place your text data in a file named `input.txt`.
3. **Configuration:** Set hyperparameters in the code (e.g., batch size, block size, number of layers, learning rate).
4. **Training:** Run the script to train the model.
5. **Generation:** Use the trained model to generate text sequences.

## Results

The model demonstrates progressive improvement in loss values during training, as indicated by periodic evaluation logs.

### Example Generated Text

After training, the model generates text based on the learned patterns:

>ENICAUTH:
I
Are chaunle therand therie
Ws withat so, planss. there;
For Henseresper Isever him's
And dither werides.

>VO: do uthend, stonds for hery imedds, sen tham isknow s hiwnVjord:
We reappn the nome feee heir swould to priyar himbliper's of I ondingly forroy: that so the fruilf.
Selave Stonges;
Burne ithapwith Boondts mow.
Wgrong thou it mor ithonk e frusust.

>ROXASCOLING:
Thene edent, by a slime ingooce,
Think, cenother,
Nowsor my she died
O, ond bas urequingnought shallararer chigh wink


Currently the generated text is still quite noisy due to limitations of the device the model was trained on.


## Future Improvements

- **Fine-tuning:** Explore different hyperparameters and architectures for better performance.
- **Dataset Variation:** Experiment with diverse datasets for training and testing.

Feel free to modify the code, tweak hyperparameters, and adapt the model for your specific text generation tasks.
