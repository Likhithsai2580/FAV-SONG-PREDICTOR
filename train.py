import os
import librosa
import numpy as np
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
from joblib import load
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import tempfile
import yt_dlp  # Replace pytubefix with yt-dlp
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    reward = 4 if action == user_feedback else -1  # +4 for correct, -1 for incorrect

    # Ensure the environment is set up
    if model.env is None:
        raise ValueError("Environment is not set up for the model.")

    # Simulate the environment step
    model.env.step([action])  # Simulate the environment step
    model.learn(total_timesteps=10, reset_num_timesteps=False)  # Update the model

    return action, reward

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
        self.user_feedback = None  # Store user feedback
    
    def reset(self):
        self.index = np.random.randint(0, len(self.data))
        return self.data[self.index]
    
    def step(self, action):
        # Calculate reward based on user feedback (passed externally)
        reward = 4 if action == self.user_feedback else -1
        
        # Update the observation space with new data (if available)
        self.index += 1
        if self.index >= len(self.data):
            self.index = 0
        
        done = False  # Episode never ends; can be modified as needed
        info = {}
        
        return self.data[self.index], reward, done, info

# Download audio from YouTube URL using yt-dlp
def download_youtube_audio(url):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path.replace('.mp3', '.%(ext)s'),  # Save with the correct extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,  # Suppress yt-dlp output
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Rename the file to have the correct extension
    downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("audio.")]
    if downloaded_files:
        os.rename(os.path.join(temp_dir, downloaded_files[0]), audio_path)

    return audio_path

# Main function
def main(youtube_url):
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

    # Download audio from YouTube URL
    audio_path = download_youtube_audio(youtube_url)

    # Extract features from the downloaded audio file
    features = extract_features(audio_path)
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

    # Set user feedback in the environment
    model.env.envs[0].user_feedback = user_feedback

    # Update the RL model with the user's feedback
    while True:
        action, reward = update_rl_model(model, features_scaled[0], user_feedback)
        new_prediction = "Favorite" if action == 1 else "Not Favorite"
        print(f"New Model Prediction: {new_prediction}, Reward: {reward}")

        if action == user_feedback:
            print("Model prediction matches user feedback! Exiting loop.")
            break

    # Save the updated model
    model.save(model_path)
    print("Model updated and saved!")

if __name__ == "__main__":
    # Provide the YouTube URL directly or via command-line argument
    youtube_url = "https://www.youtube.com/watch?v=B7xai5u_tnk"  # Replace with your YouTube URL
    main(youtube_url)