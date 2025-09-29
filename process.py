import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

file1 = 'superman.wav'
file2 = 'indiana_jones.wav'

y1, sr1 = librosa.load(file1, sr=None, mono=True)
y2, sr2 = librosa.load(file2, sr=None, mono=True)

duration = min(len(y1), len(y2))
y1, y2 = y1[:duration], y2[:duration]

def compute_fft(signal, sr):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)[:N // 2]
    freqs = np.fft.fftfreq(N, 1/sr)[:N // 2]
    return freqs, fft_magnitude

freqs1, mag1 = compute_fft(y1, sr1)
freqs2, mag2 = compute_fft(y2, sr2)

mag1_norm = mag1 / np.linalg.norm(mag1)
mag2_norm = mag2 / np.linalg.norm(mag2)

plt.figure(figsize=(14, 6))
plt.plot(freqs1, mag1_norm, label=file1)
plt.plot(freqs2, mag2_norm, label=file2, alpha=0.7)
plt.title('Frequency Spectrum Comparison (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.xlim(0, 5000)
plt.show()

length = max(len(mag1_norm), len(mag2_norm))
if len(mag1_norm) < length :
    mag1_norm = np.pad(mag1_norm, (0, length - len(mag1_norm)))
if len(mag2_norm) < length :
    mag2_norm = np.pad(mag2_norm, (0, length - len(mag2_norm)))

similarity = cosine_similarity([mag1_norm], [mag2_norm])[0][0]
print(f"Cosine Similarity between tracks: {similarity:.4f}")