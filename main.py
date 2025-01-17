import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from joblib import dump, load

# Step 1: Feature Extraction
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Calculate means and flatten
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spectral_mean = np.mean(spectral_centroid)
        
        # Combine features into a single array
        features = np.concatenate([
            mfcc_mean.flatten(),
            chroma_mean.flatten(),
            np.array([spectral_mean]).flatten()
        ])
        
        # Ensure the feature vector has exactly 27 elements
        features = np.pad(features, (0, 27 - len(features)), mode='constant')[:27]
        features = features.astype(np.float32)  # Ensure consistent dtype
        
        print(f"Features extracted: {features}")
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 2: Load all songs in the current directory
def load_songs_from_directory(directory):
    songs = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".mp3") or file_name.endswith(".wav"):
            file_path = os.path.join(directory, file_name)
            print(f"Processing: {file_name}")
            features = extract_features(file_path)
            if features is not None and len(features) == 27:
                songs.append(features)
                # Placeholder for actual labeling mechanism
                label = 1  # Should be replaced with actual labels
                labels.append(label)
    
    songs = np.array(songs, dtype=float)
    labels = np.array(labels, dtype=int)
    
    return songs, labels

# Step 3: Train a supervised model and save the scaler
def train_supervised_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler to a file
    dump(scaler, 'scaler.joblib')
    print("Scaler saved to 'scaler.joblib'")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    return model, scaler

# Step 4: Load the scaler from disk
def load_scaler():
    try:
        scaler = load('scaler.joblib')
        print("Scaler loaded from 'scaler.joblib'")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

# Step 5: Reinforcement Learning Environment
class MusicEnv(gym.Env):
    def __init__(self, data, model, scaler=None):
        super(MusicEnv, self).__init__()
        self.data = data
        self.model = model
        self.scaler = scaler if scaler is not None else load_scaler()
        if self.scaler is None:
            raise ValueError("Scaler could not be loaded.")
        self.index = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
    
    def reset(self):
        self.index = np.random.randint(0, len(self.data))
        return self.data[self.index]
    
    def step(self, action):
        # Scale the observation before passing it to the model
        scaled_observation = self.scaler.transform(np.expand_dims(self.data[self.index], axis=0))
        prediction = self.model.predict(scaled_observation)
        
        # Ensure prediction is a single scalar value
        prediction = prediction[0][0]  # Assuming the model outputs a 2D array with shape (1, 1)
        
        user_feedback = 1 if prediction > 0.5 else 0
        reward = 1 if action == user_feedback else -1
        self.index += 1
        if self.index >= len(self.data):
            self.index = 0
        done = False  # Episode never ends; can be modified as needed
        info = {}
        return self.data[self.index], reward, done, info

# Step 6: Main function
def main():
    directory = os.getcwd()
    songs, labels = load_songs_from_directory(directory)
    
    if len(songs) == 0:
        print("No valid songs found in the directory. Exiting.")
        return
    
    model, scaler = train_supervised_model(songs, labels)
    
    # Prepare data for RL
    X_scaled = scaler.transform(songs)
    
    # Create RL environment with the scaler
    env = DummyVecEnv([lambda: MusicEnv(X_scaled, model, scaler)])
    
    # Train RL model
    rl_model = PPO('MlpPolicy', env, verbose=1)
    rl_model.learn(total_timesteps=10000)
    
    # Save the RL model
    rl_model.save("music_rl_model")
    
    print("Reinforcement learning model trained and saved!")

if __name__ == "__main__":
    main()