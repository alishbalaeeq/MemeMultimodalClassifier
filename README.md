# Meme Multimodal Classifier

## Description
This project implements a multimodal deep learning model for classifying Internet memes for two primary tasks:

### Task A: Sentiment Classification
Task A involves classifying Internet memes into one of three sentiment categories: "negative," "neutral," or "positive." Sentiment classification helps determine the emotional tone of memes, which can be valuable for understanding user sentiments.

### Task B: Humor Classification
Task B focuses on identifying the type of humor expressed in Internet memes. The humor categories include "sarcastic," "humorous," "offensive," and "motivational." Memes can belong to multiple humor categories.

#### Category Mapping
- Not Humorous => 0 and Humorous (funny, very funny, hilarious) => 1
- Not Sarcastic => 0 and Sarcastic (general, twisted meaning, very twisted) => 1
- Not Offensive => 0 and Offensive (slight, very offensive, hateful offensive) => 1
- Not Motivational => 0 and Motivational => 1

## Model Architecture
The project's multimodal architecture leverages PyTorch to build a neural network capable of processing both image and text inputs. The neural network structure is organized as follows:

### Task A: Sentiment Classification
- Input layers for image and text data.
- Intermediate layers for feature extraction.
- Combination layers for merging image and text features.
- Output layer for sentiment classification (negative, neutral, positive).

### Task B: Humor Classification
- Input layers for image and text data.
- Intermediate layers for feature extraction.
- Combination layers for merging image and text features.
- Output layers for humor classification (sarcastic, humorous, offensive, motivational).
