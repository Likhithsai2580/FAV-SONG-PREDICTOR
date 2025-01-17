import os
import librosa
import numpy as np
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
from joblib import load
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained RL model
def load_rl_model(model_path, scaler):
    # Create a dummy environment for inference
    env = DummyVecEnv([lambda: MusicEnv(np.zeros((1, 27)), None, scaler)])  # Dummy data
    # Load the model and set the environment
    model = PPO.load(model_path, env=env, device='cpu')  # Force CPU usage
    return model

# Extract features from the audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spectral_mean = np.mean(spectral_centroid)
        
        features = np.concatenate([
            mfcc_mean.flatten(),
            chroma_mean.flatten(),
            np.array([spectral_mean]).flatten()
        ])
        
        features = np.pad(features, (0, 27 - len(features)), mode='constant')[:27]
        features = features.astype(np.float32)
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Ask user for feedback
def get_user_feedback():
    while True:
        feedback = input("Do you like this song? (yes/no): ").strip().lower()
        if feedback in ["yes", "no"]:
            return 1 if feedback == "yes" else 0
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

# Update the RL model with user feedback
def update_rl_model(model, features, user_feedback):
    # Simulate a step in the environment
    obs = np.array([features])
    action, _ = model.predict(obs)
    reward = 1 if action == user_feedback else -1

    # Ensure the environment is set up
    if model.env is None:
        raise ValueError("Environment is not set up for the model.")

    # Simulate the environment step
    model.env.step([action])  # Simulate the environment step
    model.learn(total_timesteps=1, reset_num_timesteps=False)  # Update the model

# Reinforcement Learning Environment
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
        prediction = self.model.predict(scaled_observation) if self.model is not None else 0.5
        
        # Ensure prediction is a single scalar value
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]  # Extract the scalar value from the array
        
        user_feedback = 1 if prediction > 0.5 else 0
        reward = 1 if action == user_feedback else -1
        self.index += 1
        if self.index >= len(self.data):
            self.index = 0
        done = False  # Episode never ends; can be modified as needed
        info = {}
        return self.data[self.index], reward, done, info

# Main function
def main(file_path):
    # Load the pre-fitted scaler
    scaler_path = "scaler.joblib"  # Path to the saved scaler
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file '{scaler_path}' not found. Please ensure the scaler is saved during training.")
        return
    scaler = load(scaler_path)

    # Load the RL model
    model_path = "music_rl_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    model = load_rl_model(model_path, scaler)

    # Extract features from the selected audio file
    features = extract_features(file_path)
    if features is None:
        print("Failed to extract features. Exiting.")
        return

    # Scale the features using the pre-fitted scaler
    features_scaled = scaler.transform([features])

    # Predict whether the song is a favorite
    action, _ = model.predict(features_scaled)
    prediction = "Favorite" if action == 1 else "Not Favorite"
    print(f"Model Prediction: {prediction}")

    # Get user feedback
    user_feedback = get_user_feedback()

    # Update the RL model with the user's feedback
    while True:
        update_rl_model(model, features_scaled[0], user_feedback)
        new_action, _ = model.predict(features_scaled)
        new_prediction = "Favorite" if new_action == 1 else "Not Favorite"
        print(f"New Model Prediction: {new_prediction}")

        if new_prediction != prediction:
            print("Model prediction changed!")
            break

    # Save the updated model
    model.save(model_path)
    print("Model updated and saved!")

if __name__ == "__main__":
    # Provide the file path directly or via command-line argument
    file_path = "hellcat.mp3"  # Replace with your audio file path
    main(file_path)