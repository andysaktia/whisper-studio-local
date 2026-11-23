# Film_Perjanjian_Lama_Lumo__The_Covenant__Indonesia
# Transkripsi faster-whisper (GPU dengan fallback CPU)

import os
import sys
import glob
import site

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def daftarkan_dll_nvidia():
    """Cari folder bin cuBLAS/cuDNN dari pip dan daftarkan ke PATH."""
    kandidat = set()

    # Semua kemungkinan lokasi site-packages (termasuk venv)
    roots = list(sys.path)
    try:
        roots += site.getsitepackages()
    except Exception:
        pass
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    for root in roots:
        for lib in ("cublas", "cudnn"):
            kandidat.add(os.path.join(root, "nvidia", lib, "bin"))

    ditemukan = []
    for bin_dir in sorted(kandidat):
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
            ditemukan.append(bin_dir)
    return ditemukan


folder_dll = daftarkan_dll_nvidia()
ada_cublas = any(glob.glob(os.path.join(d, "cublas64_12.dll")) for d in folder_dll)

from faster_whisper import WhisperModel
from pathlib import Path

# ==================== PENGATURAN ====================
audio_file = "02_Penciptaan_Manusia.wav"   # Ganti dengan nama file kamu
model_size = "large-v3-turbo"              # Alternatif: large-v3, medium, small
bahasa = "id"
# ====================================================

# Pilih device: GPU kalau DLL CUDA ditemukan, kalau tidak CPU
if ada_cublas:
    device, compute_type = "cuda", "float16"
else:
    print("PERINGATAN: cublas64_12.dll tidak ditemukan, memakai CPU.")
    print("Untuk GPU: pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"\n")
    device, compute_type = "cpu", "int8"


def buat_model(device, compute_type):
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transkripsi(model):
    segments, info = model.transcribe(
        audio_file,
        language=bahasa,
        beam_size=5,
        vad_filter=True,
        log_progress=True,
    )

    print(f"\nBahasa terdeteksi : {info.language}")
    print(f"Probabilitas      : {info.language_probability:.2f}")
    print(f"Durasi audio      : {info.duration:.2f} detik")
    print("-" * 60)

    output_txt = Path(audio_file).with_suffix(".txt")
    output_srt = Path(audio_file).with_suffix(".srt")

    with open(output_txt, "w", encoding="utf-8") as f_txt, \
         open(output_srt, "w", encoding="utf-8") as f_srt:

        for i, seg in enumerate(segments, 1):
            text = seg.text.strip()
            print(f"[{seg.start:7.2f}s → {seg.end:7.2f}s] {text}")

            f_txt.write(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}\n")

            f_srt.write(f"{i}\n")
            f_srt.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f_srt.write(f"{text}\n\n")

    return output_txt, output_srt


if not Path(audio_file).exists():
    sys.exit(f"File audio tidak ditemukan: {audio_file}")

print(f"Memuat model ({model_size}) di {device}...")
model = buat_model(device, compute_type)
print("Model berhasil dimuat!\n")
print(f"Mulai transkripsi: {audio_file}\n")

try:
    hasil = transkripsi(model)
except RuntimeError as e:
    # Error CUDA/cuBLAS/cuDNN saat inferensi -> ulangi di CPU
    if device == "cuda":
        print(f"\nGPU gagal ({e}).")
        print("Mengulang dengan CPU (int8)...\n")
        del model
        model = buat_model("cpu", "int8")
        hasil = transkripsi(model)
    else:
        raise

print("\n" + "=" * 60)
print("Selesai!")
print("Hasil disimpan ke:")
print(f"  - {hasil[0]}")
print(f"  - {hasil[1]}")