import numpy as np
import sounddevice as sd
import librosa
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time
from collections import deque
import random

# Set parameters
fs = 44100  # Sampling frequency (standard for audio)
frame_size = 512  # Number of frames (512 samples = ~11.6 ms at 44.1kHz)
num_colors = 16  # Number of colors to store in the queue

# Queue to store concatenated hex color strings
color_queue = deque(maxlen=num_colors)

# To store scatter plot data
scatter_data = []

# Function to convert audio features to HSV-based color in HEX
def audio_to_color(audio_data):
    # Calculate audio features: spectral centroid, ZCR, spectral contrast, RMS
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=fs, n_fft=frame_size).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=fs, n_fft=frame_size).mean()
    rms = librosa.feature.rms(y=audio_data).mean()

    rms = librosa.feature.rms(y=audio_data).mean()
    spectral_contrast = 1000 * (spectral_contrast - int(spectral_contrast))
    zero_crossing_rate *= 100  # Convert ZCR to percentage
    zero_crossing_rate = 1000 * (zero_crossing_rate - int(zero_crossing_rate))
    spectral_centroid = 1000 * (spectral_centroid - int(spectral_centroid))
    rms = 1000000 * (rms - int(rms))
    # Modify HSV mapping: combining multiple features
    redc = np.clip((spectral_centroid + rms) * 256 / 2000, 0, 255)  # Combine centroid and RMS for hue
    greenc = np.clip((zero_crossing_rate + spectral_contrast) * 256 / 2000, 0, 255)  # Combine ZCR and contrast for saturation
    bluec = np.clip(spectral_contrast * 256 / 1000, 0, 255)  # Combine RMS and contrast for value

    # Normalize RGB values to [0, 1] range
    rgb_color = (redc / 255, greenc / 255, bluec / 255)

    # Convert RGB to HEX
    hex_color = mcolors.to_hex(rgb_color)[1:]  # Remove the "#" prefix

    return hex_color

# Function to get current timestamp and queue state
def get_color_queue_state():
    current_timestamp = time.time()  # Get current time
    # Concatenate all the hex colors into a single string
    concatenated_colors = ''.join(color_queue)
    return current_timestamp, concatenated_colors

# Callback function for real-time audio input
i = 0
def audio_callback(indata, frames, time, status):
    global i
    if status:
        print(status)

    # Process the audio frame (mono)
    audio_data = indata[:, 0]

    # Convert the audio frame to a color in HEX (without "#")
    hex_color = audio_to_color(audio_data)

    # Convert HEX color back to RGB for plotting
    rgb_color = mcolors.hex2color('#' + hex_color)

    # Generate a random alpha value between 0.3 (transparent) and 1 (opaque)
    alpha = random.uniform(0.3, 1.0)

    i += 1

    # Append data to scatter_data list for plotting after the recording
    scatter_data.append((i, random.randint(0, 1000), rgb_color, alpha))

    # Append the HEX color to the queue
    color_queue.append(hex_color)

    # Get and print the current concatenated hex color string and timestamp
    timestamp, current_colors = get_color_queue_state()
    print(f"Timestamp: {timestamp}, TheRandomNumber: {current_colors}")

# Main function to run real-time audio processing
def run_real_time_processing():
    try:
        print("Recording... Press Ctrl+C to stop.")
        # Open an input stream for real-time audio
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=frame_size):
            while True:
                time.sleep(0.1)  # Simulate real-time processing
    except KeyboardInterrupt:
        print("Recording stopped.")
        plot_scatter()  # Call the plot function after recording
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to plot the scatter data after processing
def plot_scatter():
    for x, y, color, alpha in scatter_data:
        rgba_color = (*color, alpha)  # Combine RGB with alpha
        plt.scatter(x, y, color=rgba_color, marker="o")
    plt.title("Scatter Plot of RGB Colors with Varying Opacity")
    plt.xlabel("Index")
    plt.ylabel("Random Y")
    plt.show()

# Run the program
if __name__ == "__main__":
    run_real_time_processing()
