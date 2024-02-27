import tkinter as tk
from tkinter import filedialog
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Function to perform tone emotion detection from an audio file
def audio_file_tone_emotion_detection(file_path, chunk_size=1024):
    try:
        audio_data, rate = librosa.load(file_path, sr=None, mono=True)

        plt.ion()
        fig, ax = plt.subplots(2, 1)
        x = np.arange(0, len(audio_data) / rate, 1 / rate)
        line, = ax[0].plot(x, audio_data)
        ax[0].set_title('Audio Signal')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlim(0, len(audio_data) / rate)
        ax[0].grid(True)

        # Define emotions
        emotions = ['Neutral', 'Joy', 'Anger', 'Sad', 'Analytical', 'Confident', 'Surprise', 'Fear']
        # Bar plot settings
        bar_width = 0.35
        index = np.arange(len(emotions))
        rects = ax[1].bar(index, [0] * len(emotions), bar_width, label='Emotion')
        ax[1].set_title('Detected Emotion')
        ax[1].set_xlabel('Emotion')
        ax[1].set_ylabel('Probability')
        ax[1].set_xticks(index + bar_width / 2)
        ax[1].set_xticklabels(emotions)
        ax[1].set_ylim(0, 1)
        ax[1].grid(True)

        fig.tight_layout()

        print("Starting audio file tone emotion detection...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]

            # Perform emotion detection based on the audio data
            emotion_probabilities = detect_emotion(chunk)

            # Update emotion bar plot
            for rect, prob in zip(rects, emotion_probabilities):
                rect.set_height(prob)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Insert a small delay to simulate real-time processing
            time.sleep(0.1)
    except Exception as e:
        print("Error during audio file tone emotion detection:", e)

# Function to perform tone emotion detection based on audio data
def detect_emotion(audio_data):
    # Example simple rule-based approach:
    audio_feature = np.mean(audio_data)
    if audio_feature > 0.1:
        return [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Joy
    elif audio_feature < -0.1:
        return [0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Anger
    elif audio_feature < -0.05:
        return [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]  # Sad
    elif audio_feature > 0.05:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]  # Surprise
    elif audio_feature < -0.15:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]  # Fear
    elif audio_feature > 0.2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]  # Confident
    else:
        return [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0]  # Neutral
def start_detection(file_path_label):
    file_path = file_path_label.cget("text")
    if file_path:
        threading.Thread(target=audio_file_tone_emotion_detection, args=(file_path,)).start()

def browse_file(file_path_label):
    file_path = filedialog.askopenfilename()
    file_path_label.config(text=file_path)

# GUI setup
root = tk.Tk()
root.title("Speech Emotion Detection")
root.geometry('500x200')  # Increase the size of the window

file_path_label = tk.Label(root, text="")
file_path_label.pack(side=tk.TOP, pady=20)  # Placing the label in the middle with padding

browse_button = tk.Button(root, text="Browse", command=lambda: browse_file(file_path_label), padx=10, pady=10)
browse_button.pack(side=tk.BOTTOM, pady=(0, 10))

start_button = tk.Button(root, text="Start Detection", command=lambda: start_detection(file_path_label), padx=10, pady=10)
start_button.pack(side=tk.BOTTOM)

root.mainloop()

