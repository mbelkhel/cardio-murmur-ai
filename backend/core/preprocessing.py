import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Audio Settings ---
TARGET_SR = 4000

def preprocess_data(raw_data_dir, window_size_sec=2, overlap_sec=1, target_sr=TARGET_SR):
    """
    Processes audio files from raw_data_dir, applying windowing and CQT.

    Args:
        raw_data_dir (str): Path to the directory containing .wav files.
        window_size_sec (float): Desired window size in seconds.
        overlap_sec (float): Desired overlap between windows in seconds.
        target_sr (int): Target sample rate for audio loading.

    Returns:
        tuple: A tuple containing three lists:
               - X (list): A list of processed CQT spectrograms (as numpy arrays).
               - y_labels (list): A list of corresponding labels for each spectrogram.
               - file_identifiers (list): A list of source file basenames for each window.
    """
    X = []
    y_labels = []
    file_identifiers = [] # New list

    print(f"Starting preprocessing with windowing from: {raw_data_dir}")
    print(f"Window size: {window_size_sec}s, Overlap: {overlap_sec}s, Target SR: {target_sr}Hz")

    window_samples = int(target_sr * window_size_sec)
    overlap_samples = int(target_sr * overlap_sec)
    step_samples = window_samples - overlap_samples

    if step_samples <= 0:
        raise ValueError("Overlap results in zero or negative step size. Please reduce overlap or increase window size.")

    file_metadata = []
    # Ensure raw_data_dir exists before listing its contents
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return X, y_labels

    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.wav'):
            parts = filename.replace('.wav', '').split('_')
            if len(parts) == 4:
                disease, patient_id, position, area = parts
                file_path = os.path.join(raw_data_dir, filename)
                file_metadata.append({
                    'filepath': file_path,
                    'disease_class': disease,
                    'patient_id': patient_id,
                    'position': position,
                    'area': area
                })
            else:
                print(f"Skipping file with unexpected name format: {filename}")

    files_df = pd.DataFrame(file_metadata)
    if files_df.empty:
        print(f"No .wav files found with expected naming format or metadata could not be extracted from {raw_data_dir}")
        return X, y_labels

    for index, row in tqdm(files_df.iterrows(), total=files_df.shape[0], desc="Processing files"):
        filepath = row['filepath']
        label = row['disease_class']

        try:
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

            # Placeholder for windowing and CQT logic (to be implemented in next steps)
            if len(audio) >= window_samples:
                for start_sample in range(0, len(audio) - window_samples + 1, step_samples):
                    window = audio[start_sample : start_sample + window_samples]

                    # --- SAFETY CHECK for silent/very short window (though window_samples should ensure length) ---
                    if np.max(np.abs(window)) == 0:
                        # print(f"Skipping silent window in file: {os.path.basename(filepath)}") # Optional: too verbose
                        continue

                    # 1. Create the Constant-Q Transform
                    # Ensure 'sr' from librosa.load (which is target_sr) is used.
                    # Parameters from the notebook: fmin=librosa.note_to_hz('C1'), n_bins=60
                    cqt = librosa.cqt(y=window, sr=target_sr,
                                      fmin=librosa.note_to_hz('C1'),
                                      n_bins=60)

                    cqt_db = librosa.amplitude_to_db(np.abs(cqt))

                    # 2. Normalize the CQT image (using the same function as in the notebook)
                    # S.min() and S.max() can be zero if cqt_db is all zeros (e.g. silent window after db conversion)
                    # Add a small epsilon to prevent division by zero if (S.max() - S.min()) is zero.
                    min_val = cqt_db.min()
                    max_val = cqt_db.max()
                    if (max_val - min_val) == 0:
                        # This case means the window is silent or constant value after CQT and dB conversion.
                        # Resulting spectrogram will be all zeros (or constant).
                        # print(f"Window in {os.path.basename(filepath)} has constant CQT values after dB conversion. Result will be zeros.") # Optional
                        final_cqt = np.zeros_like(cqt_db) # Or handle as per desired logic, e.g., skip
                    else:
                        final_cqt = (cqt_db - min_val) / (max_val - min_val + 1e-6) # Epsilon added for safety

                    X.append(final_cqt)
                    y_labels.append(label)
                    file_identifiers.append(os.path.basename(filepath)) # Add this line
            else:
                print(f"Skipping file {os.path.basename(filepath)}: shorter than window size ({len(audio)/sr:.2f}s < {window_size_sec}s).")

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    print(f"Preprocessing with CQT and normalization complete. X contains processed spectrograms.")
    return X, y_labels, file_identifiers

if __name__ == '__main__':
    print("Running preprocess_data in standalone mode for testing structure...")

    # This path assumes execution from the project root directory.
    sample_data_dir = 'model_training/data/murmur_dataset_v1'

    # Check if the directory exists from the perspective of the script's location if run directly
    # This is more for direct execution testing. Subtasks usually run from root.
    if not os.path.exists(sample_data_dir):
         # Try path relative to script location for direct execution
        alt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_training', 'data', 'murmur_dataset_v1')
        if os.path.exists(alt_path):
            sample_data_dir = alt_path
        else:
            print(f"Sample data directory not found at {os.path.abspath(sample_data_dir)} or {os.path.abspath(alt_path)}")
            print("Please ensure the path is correct.")
            # Exiting if no data, as the test is pointless
            exit()

    print(f"Attempting to process data from: {os.path.abspath(sample_data_dir)}")
    X_processed, y_processed, ids_processed = preprocess_data(sample_data_dir)
    print(f"Example usage finished. Number of CQT spectrogram windows generated: {len(X_processed)}, IDs: {len(ids_processed)}")
    if ids_processed:
        print(f"Example file identifier for first window: {ids_processed[0]}")
