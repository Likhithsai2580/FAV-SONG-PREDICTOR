# FAV-SONG-PREDICTOR

## Project Description

FAV-SONG-PREDICTOR is a machine learning project designed to predict whether a song is a user's favorite based on its audio features. The project utilizes both supervised and reinforcement learning models to achieve this goal.

## Features

- Extracts audio features such as MFCC, chroma, and spectral centroid from songs.
- Trains a supervised learning model to predict whether a song is a favorite.
- Utilizes reinforcement learning to improve predictions based on user feedback.
- Provides a command-line interface for users to interact with the model and provide feedback.

## Installation

To install the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Likhithsai2580/FAV-SONG-PREDICTOR.git
   cd FAV-SONG-PREDICTOR
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, follow these steps:

1. Ensure you have some audio files (e.g., MP3 or WAV) in the project directory.

2. Run the main script to train the models:
   ```bash
   python main.py
   ```

3. Use the `use.py` script to interact with the model and provide feedback:
   ```bash
   python use.py
   ```

4. Use the `train.py` script to update the reinforcement learning model with user feedback:
   ```bash
   python train.py
   ```

## Contribution Guidelines

We welcome contributions to improve the project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request to the main repository.

## Supervised and Reinforcement Learning Models

### Supervised Learning Model

The supervised learning model is trained using audio features extracted from songs. It uses a neural network with dense layers to predict whether a song is a favorite based on these features. The model is trained using labeled data, where each song is labeled as a favorite or not.

### Reinforcement Learning Model

The reinforcement learning model is designed to improve predictions based on user feedback. It uses the PPO (Proximal Policy Optimization) algorithm to learn from user interactions. The model is trained in a custom environment where it receives rewards based on the accuracy of its predictions and user feedback.
