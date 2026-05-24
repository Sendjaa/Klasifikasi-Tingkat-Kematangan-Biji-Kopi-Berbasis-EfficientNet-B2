# =====================================================================
#  app.py — GUI Deteksi Kopi + Grad-CAM + Dynamic Position Clustering (Crop Mode)
# =====================================================================

import os
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
from sklearn.decomposition import PCA

import config
from model import load_model

# 1. Konfigurasi Device & Load Model
device = config.DEVICE
print(f"[*] Memuat model untuk aplikasi Gradio di {device.upper()}...")
model = load_model(str(config.BEST_MODEL_PATH), device=device)

# Ambil Feature Extractor
if hasattr(model, 'features'):
    feature_extractor = model.features
else:
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
model.eval()

# 2. Transformasi Gambar (Tanpa CenterCrop agar mengikuti hasil crop user)
def preprocess_cropped_image(image_pil):
    img_rgb = image_pil.convert("RGB")
    # Langsung resize ke 224x224 dari hasil potongan selektif user
    img_resized = transforms.functional.resize(img_rgb, (224, 224))
    return img_resized

tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Metadata Warna
ROAST_COLOR_MAP = {
    "light": {"hex": "#CD7F32", "desc": "Cokelat Terang / Cinnamon Roast (Asam Tinggi)"},
    "medium": {"hex": "#8B5A2B", "desc": "Cokelat Sedang / City Roast (Seimbang)"},
    "dark": {"hex": "#3B2219", "desc": "Cokelat Sangat Gelap / French Roast (Pahit Dominan)"}
}

# 3. GLOBAL FITUR: Mengunci Matriks Proyeksi PCA Permanen (Anti-Acak)
print("[*] Mengunci matriks proyeksi PCA dari dataset uji untuk sinkronisasi posisi...")
test_dir = os.path.join(config.DATA_DIR, 'test')

features_all = []
labels_all = []
class_to_idx = {}

if os.path.exists(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    class_to_idx = test_dataset.class_to_idx
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = feature_extractor(images)
            if len(feats.shape) == 4:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)
            features_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())
            
    features_all = np.concatenate(features_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
else:
    features_all = np.random.randn(30, 1280)
    labels_all = np.array([0]*10 + [1]*10 + [2]*10)
    class_to_idx = {"light": 0, "medium": 1, "dark": 2}

# Fit PCA baseline sekali saja biar koordinat kluster tidak berubah-ubah
pca_transformer = PCA(n_components=2, random_state=42)
test_coords = pca_transformer.fit_transform(features_all)

# 4. Fungsi Grad-CAM Sederhana
def generate_gradcam(image_cropped, model, target_layer):
    image_tensor = tensor_normalize(image_cropped).unsqueeze(0).to(device)
    image_tensor.requires_grad_()
    
    feature_maps, gradients = [], []
    def forward_hook(module, input, output): feature_maps.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])
        
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)
    
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=-1).detach()
    idx = torch.argmax(probs, dim=-1).item()
    
    model.zero_grad()
    logits[0, idx].backward()
    h_forward.remove()
    h_backward.remove()
    
    grads = gradients[0].detach().cpu().numpy()[0]
    f_maps = feature_maps[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2)) if len(grads.shape) == 3 else grads
    
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32) if len(f_maps.shape) > 2 else np.zeros(f_maps.shape, dtype=np.float32)
    if len(f_maps.shape) == 3:
        for i, w in enumerate(weights): cam += w * f_maps[i]
    else: cam = f_maps
        
    cam = np.maximum(cam, 0)
    if np.max(cam) != 0: cam = cam / np.max(cam)
    cam = cv2.resize(np.squeeze(cam), (224, 224))
    
    img_np = cv2.cvtColor(np.array(image_cropped), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result_cam = cv2.cvtColor(cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0), cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_cam), probs.cpu().numpy()[0]

# 5. Fungsi Menggambar Scatter Plot Dinamis (PCA-Based)
def draw_dynamic_pca(new_feature):
    new_img_coord = pca_transformer.transform(new_feature)[0]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    
    custom_colors = {
        "light": "#CD7F32",
        "medium": "#8B5A2B",
        "dark": "#3B2219"
    }
    
    # Plot data kluster lama
    for class_name, class_idx in class_to_idx.items():
        if class_name in ['medium_dark', '.ipynb_checkpoints']: continue
        indices = np.where(labels_all == class_idx)
        color = custom_colors.get(class_name.lower(), "#7f8c8d")
        ax.scatter(test_coords[indices, 0], test_coords[indices, 1], c=color, label=class_name.capitalize(), s=25, alpha=0.35, edgecolors='none')
    
    # Plot posisi koordinat kopi hasil crop (🌟 Bintang Merah)
    ax.scatter(new_img_coord[0], new_img_coord[1], c='#ff1e1e', marker='*', s=200, label='Kopi Terpilih', edgecolors='black', linewidths=1.2, zorder=10)
    
    ax.set_xlabel('Principal Component 1', fontsize=9)
    ax.set_ylabel('Principal Component 2', fontsize=9)
    ax.set_title('Letak Posisi Kopi Terpilih pada Latent Space', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_plot = Image.frombuffer('RGBA', fig.canvas.get_width_height(), rgba_buffer, 'raw', 'RGBA', 0, 1).convert('RGB')
    plt.close(fig)
    return img_plot

# 6. Fungsi Utama Prediksi Interface
def predict_and_visualize(image_pil):
    if image_pil is None:
        return None, "Silakan masukkan gambar sampel kopi.", "<div style='padding:15px; background-color:#2a2a2a; text-align:center; color:#888;'>Menunggu gambar...</div>", None
    
    class_names = config.CLASSES
    # Menggunakan prapemrosesan dinamis tanpa center crop paksaan
    image_cropped = preprocess_cropped_image(image_pil)
    
    image_tensor = tensor_normalize(image_cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        new_feat = feature_extractor(image_tensor)
        if len(new_feat.shape) == 4:
            new_feat = torch.nn.functional.adaptive_avg_pool2d(new_feat, (1, 1))
        new_feat = new_feat.view(new_feat.size(0), -1).cpu().numpy()
    
    try:
        target_layer = model.features[-1] if hasattr(model, 'features') else [m for m in list(model.modules()) if isinstance(m, torch.nn.Conv2d)][-1]
        cam_image, probs = generate_gradcam(image_cropped, model, target_layer)
    except Exception as e:
        print(f"[!] Fallback Grad-CAM aktif: {e}")
        with torch.no_grad(): probs = F.softmax(model(image_tensor), dim=-1).cpu().numpy()[0]
        cam_image = image_cropped

    results = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    if 'medium_dark' in results: del results['medium_dark']
        
    highest_class = max(results, key=results.get)
    highest_prob = results[highest_class] * 100
    color_meta = ROAST_COLOR_MAP.get(highest_class, {"hex": "#333333", "desc": "Unknown"})
    
    tsne_dynamic_plot = draw_dynamic_pca(new_feat)
    
    classification_status_html = f"""
    <div style="background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; font-family: sans-serif; box-shadow: 2px 2px 12px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 5px;">
            <span style="height: 20px; width: 20px; background-color: {color_meta['hex']}; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px {color_meta['hex']};"></span>
            <span style="font-size: 22px; font-weight: bold; color: #ffffff; letter-spacing: 1px;">{highest_class.upper()} ROAST</span>
        </div>
        <div style="text-align: center; color: #aaa; font-size: 13px;">Confidence Score: <b style="color: #ffb142;">{highest_prob:.2f}%</b></div>
        <hr style="border: 0; border-top: 1px solid #444; margin: 10px 0;">
        <div style="text-align: center; color: #888; font-size: 12px; font-style: italic;">{color_meta['desc']}</div>
    </div>
    """
    
    info_text = f"Analisis spasial selesai berdasarkan area biji kopi yang kamu crop secara spesifik."
    return cam_image, info_text, classification_status_html, tsne_dynamic_plot

# 7. Antarmuka Layout Gradio (Mengaktifkan Tool Crop Interaktif)
# 7. Antarmuka Layout Gradio (Fixed Version untuk Gradio Terbaru)
with gr.Blocks() as demo:
    gr.Markdown("# ☕ Coffee Roast Level Analyzer (XAI Premium Edition)")
    
    with gr.Row():
        with gr.Column():
            # Menggunakan interactive=True agar tool crop bawaan Gradio otomatis muncul di pojok gambar
            input_img = gr.Image(type="pil", label="1. Unggah Foto & Crop Biji Kopi Spesifik", interactive=True)
            gr.Markdown("<small>*Tips: Setelah unggah gambar grid, gunakan tombol crop (ikon kotak/pensil) pada gambar untuk menyeleksi satu area biji kopi spesifik agar letak koordinatnya akurat.</small>")
            btn_submit = gr.Button("Mulai Analisis Komponen Kopi", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🎯 2. Hasil Klasifikasi")
            output_status = gr.HTML(value="<div style='padding:15px; border-radius:8px; background-color:#2a2a2a; text-align:center; color:#888;'>Belum ada analisis dilakukan</div>")
            
            with gr.Tabs():
                with gr.TabItem("📊 Global Analysis (Grad-CAM)"):
                    output_cam = gr.Image(type="pil", label="Peta Fokus Tingkat Kematangan Warna")
                    
                with gr.TabItem("📈 Letak Posisi Kopi (Space Clustering)"):
                    gr.Markdown("Letak posisi koordinat biji kopi yang kamu potong/pilih saat ini (ditandai **Bintang Merah 🌟**):")
                    output_tsne = gr.Image(type="pil", label="Visualisasi Titik Lokasi Kopi")
                    
                with gr.TabItem("💡 Detailed Info"):
                    output_info = gr.Douglas = gr.Textbox(label="Keterangan Analisis", lines=3, interactive=False)
            
    btn_submit.click(
        fn=predict_and_visualize, 
        inputs=input_img, 
        outputs=[output_cam, output_info, output_status, output_tsne]
    )

if __name__ == "__main__":
    demo.launch(share=False)