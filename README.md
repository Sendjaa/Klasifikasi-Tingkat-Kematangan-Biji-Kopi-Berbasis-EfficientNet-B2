Coffee Roast Level Analyzer (XAI Premium Edition)
Proyek ini adalah sistem Explainable AI (XAI) berbasis Deep Learning yang dirancang untuk mendeteksi tingkat kematangan biji kopi secara otomatis. Dengan memanfaatkan arsitektur EfficientNet-B2 dan teknik visualisasi feature space, sistem ini memberikan transparansi mengenai bagaimana model mengambil keputusan klasifikasi.

🚀 Fitur Utama
Klasifikasi Kopi Otomatis: Menggunakan model fine-tuned EfficientNet-B2 untuk membedakan kelas Light, Medium, dan Dark Roast.

Interpretasi Heatmap (Grad-CAM): Visualisasi area piksel yang menjadi fokus utama model saat menentukan tingkat kematangan kopi.

Analisis Ruang Laten (PCA Clustering): Visualisasi real-time posisi biji kopi yang diuji dalam ruang fitur model. Titik data individu dipetakan untuk membandingkan karakteristiknya dengan dataset referensi.

Fitur Crop Interaktif: Antarmuka berbasis Gradio yang memungkinkan pengguna menyeleksi (crop) biji kopi spesifik untuk analisis fitur yang lebih presisi.

🛠️ Tech Stack
Language: Python 3.11

Deep Learning: PyTorch, Torchvision

XAI & Math: OpenCV (Grad-CAM), Matplotlib, Scikit-learn (PCA)

GUI: Gradio

📂 Struktur Direktori
Plaintext
deteksi_kopi/
├── data/               # Dataset (train, test, val)
├── models/             # best_model.pth
├── results/            # Hasil visualisasi
├── app.py              # Aplikasi Gradio
├── config.py           # Konfigurasi parameter
├── model.py            # Arsitektur model
└── README.md           # Dokumentasi ini
⚙️ Cara Menjalankan
Install dependensi:

Bash
pip install -r requirements.txt
Jalankan Aplikasi:
Pastikan file model best_model.pth sudah tersedia di folder /models, lalu jalankan:

Bash
py app.py
Akses Dashboard:
Buka browser dan arahkan ke alamat http://127.0.0.1:7860.

Penggunaan:

Unggah foto biji kopi.

Gunakan tool Crop (ikon kotak pada gambar) untuk memilih biji kopi yang ingin dianalisis.

Klik "Mulai Analisis Komponen Kopi" untuk melihat hasil klasifikasi dan posisi spasialnya dalam ruang laten.

📝 Catatan
Proyek ini dikembangkan untuk memberikan transparansi pada sistem Black Box dalam klasifikasi visual, menjadikannya alat yang akurat dan dapat dipercaya dalam analisis kualitas biji kopi.