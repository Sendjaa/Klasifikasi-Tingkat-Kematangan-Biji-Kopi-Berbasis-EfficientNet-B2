import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

import config
from model import load_model

def main():
    device = config.DEVICE
    print(f"[*] Memuat model untuk ekstraksi fitur t-SNE di {device.upper()}...")
    model = load_model(str(config.BEST_MODEL_PATH), device=device)
    
    # Mengambil layer feature extractor sebelum classifier terakhir
    if hasattr(model, 'features'):
        feature_extractor = model.features
    else:
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    feature_extractor.eval()

    # Transformasi standar gambar masuk model
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Jalur folder test (diambil dari config.py)
    test_dir = os.path.join(config.DATA_DIR, 'test')
    if not os.path.exists(test_dir):
        print(f"[!] Folder test tidak ditemukan di: {test_dir}")
        print("[!] Pastikan path DATA_DIR di config.py mengarah ke direktori data yang benar.")
        return

    print(f"[*] Membaca data uji menggunakan ImageFolder dari: {test_dir}")
    # Menggunakan datasets.ImageFolder sebagai pengganti CoffeeDataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    features_list = []
    labels_list = []

    print("[*] Mengekstrak representasi fitur spasial dari data uji...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            features = feature_extractor(images)
            if len(features.shape) == 4:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    # ... (Proses algoritma t-SNE di atas tetap sama) ...
    print("[*] Menjalankan algoritma t-SNE (Mereduksi dimensi vektor ke 2D)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, max_iter=1000)
    features_2d = tsne.fit_transform(features_all)

    # 🌟 PERBAIKAN DINAMIS UNTUK MENGATASI INDEXERROR
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Mapping nama kelas ke warna premium custom milikmu
    custom_colors = {
        "light": "#CD7F32",
        "medium": "#8B5A2B",
        "dark": "#3B2219"
    }
    
    # Gunakan colormap default matplotlib (tab10) sebagai cadangan jika ada kelas tak terduga
    backup_cmap = plt.get_cmap('tab10')

    print("[*] Membuat grafik scatter plot...")
    for class_idx, class_name in enumerate(test_dataset.classes):
        # Jika folder yang dibaca adalah 'medium_dark' atau folder checkpoint, kita skip saja
        if class_name in ['medium_dark', '.ipynb_checkpoints']:
            print(f"[!] Mengabaikan kelas tambahan: '{class_name}' agar tidak merusak visualisasi 3 kelas.")
            continue
            
        indices = np.where(labels_all == class_idx)
        
        # Ambil warna dari map custom, kalau tidak ada pakai warna cadangan matplotlib
        color = custom_colors.get(class_name.lower(), backup_cmap(class_idx % 10))
        
        plt.scatter(
            features_2d[indices, 0], 
            features_2d[indices, 1], 
            c=color, 
            label=f"{class_idx}: {class_name.capitalize()}",
            edgecolors='none',
            s=40,
            alpha=0.8
        )

    plt.xlabel('Dimension 1 (t-SNE)', fontsize=11)
    plt.ylabel('Dimension 2 (t-SNE)', fontsize=11)
    plt.title('Latent Feature Space Clustering (EfficientNet-B2)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#ddd')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Simpan hasil plotting
    output_path = 'results/tsne_clustering.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Grafik t-SNE sukses dibuat dan disimpan di: {output_path}")

if __name__ == "__main__":
    main()