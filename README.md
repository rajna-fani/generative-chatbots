# Chatbot Experimentation README

## Overview
This project explores the implementation and evaluation of advanced neural network models for generative chatbot systems. It focuses on training Sequence-to-Sequence models, a Transformer encoder-decoder model, and a generative pre-trained model to engage in conversational dialogue.
The project was carried out prior to the surge in popularity of ChatGPT and LLMs, hence it utilizes earlier model architectures.

## Models
- **Sequence-to-Sequence Models**: Includes five models with different configurations:
  - Three models based on bidirectional GRU (Gated Recurrent Unit) cells.
  - Two models employing unidirectional LSTM (Long Short-Term Memory) cells.
- **Transformer Model**: Utilizes an encoder-decoder architecture tailored for conversational applications.
- **Generative Pre-trained Model**: Leverages a model pre-trained by Microsoft for generative dialogue tasks.

## Datasets
Two conversational datasets were used for training the models:
- **Cornell Movie-Dialogue Corpus**: Contains over 220,579 conversational exchanges between movie characters, extracted from 617 movies.
- **DailyDialog Dataset**: A manually labeled, multi-turn dialogue dataset covering various topics reflective of daily communication.

## Preprocessing
The preprocessing steps involved cleaning the datasets, creating source-target sentence pairs, and preparing the data for encoder input. This included removing unnecessary metadata, lowercasing text, and filtering by utterance length.

## Training Details
- The models were implemented using PyTorch.
- Sequence-to-Sequence models and the Transformer model were trained with different batch sizes and iterations/epochs to optimize performance.
- Training time ranged from 3 to 15 hours depending on the model and dataset, using an M1 Pro Chip with a 10-Core CPU and 16-Core Neural Engine.

## Evaluation
Models were evaluated based on qualitative and quantitative analyses, including BLEU score calculations and average loss measurements. The evaluation highlighted the performance of each model type and the impact of different configurations on the generative capabilities of the chatbots.

## Conclusion
The experiments conducted offer insights into the complexities of training neural network-based chatbots and underscore the importance of model architecture, dataset choice, and training parameters in achieving desired conversational capabilities.

