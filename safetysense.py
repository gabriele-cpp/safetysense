# Cautions : Kode dijalankan secara bertahap agar model bisa membaca dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# path dataset (bervariasi tergantung lokasi dataset di local computer)
path = "E:/UCI HAR Dataset/UCI HAR Dataset/"

# load data fitur x
X_train = pd.read_csv(path + 'train/X_train.txt', sep='\s+', header=None)
X_test = pd.read_csv(path + 'test/X_test.txt', sep='\s+', header=None)

# load data label y
y_train = pd.read_csv(path + 'train/y_train.txt', sep='\s+', header=None)
y_test = pd.read_csv(path + 'test/y_test.txt', sep='\s+', header=None)

# Memuat daftar nama fitur (kolom)
feature_names = pd.read_csv(path + 'features.txt', sep='\s+', header=None)
feature_names = feature_names[1] # Ambil kolom kedua (hanya nama fiturnya)

# Terapkan nama fitur ke data X_train dan X_test
X_train.columns = feature_names
X_test.columns = feature_names

# Kata kunci fitur yang ingin kita pertahankan (mean, std, gravity)
keywords = ['mean()', 'std()', 'gravity']

# Filter nama fitur: pilih kolom yang mengandung salah satu keyword di atas
fitur_terpilih = [col for col in X_train.columns if any(keyword in col for keyword in keywords)]

# Feature selection
# Jumlah fitur setelah reduksi (seharusnya berkurang drastis dari 561)
print(f"Jumlah Fitur Awal: 561")
print(f"Jumlah Fitur Setelah Reduksi: {len(fitur_terpilih)}")

# Terapkan filter ke dataset X_train dan X_test
X_train_reduced = X_train[fitur_terpilih]
X_test_reduced = X_test[fitur_terpilih]

print(f"\nBentuk Data Training Baru: {X_train_reduced.shape}")
print(f"Bentuk Data Testing Baru: {X_test_reduced.shape}")

# 1. Membuat Model Random Forest
model_reduced = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Melatih Model (Training)
print("\nMulai melatih model dengan fitur yang dikurangi...")
# Pastikan y_train tetap dalam format 1D (flat)
model_reduced.fit(X_train_reduced, y_train.values.ravel())
print("Pelatihan selesai!")

# 3. Prediksi dan Evaluasi
y_pred_reduced = model_reduced.predict(X_test_reduced)
akurasi_reduced = accuracy_score(y_test, y_pred_reduced)

print(f"\nAkurasi Prediksi dengan Fitur yang Diringankan: {akurasi_reduced*100:.2f}%")




