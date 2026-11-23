# Transkripsi batch faster-whisper (GPU dengan fallback CPU)
# Film Perjanjian Lama Lumo "The Covenant" Indonesia

import os
import sys
import glob
import site
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def daftarkan_dll_nvidia():
    """Cari folder bin cuBLAS/cuDNN dari pip dan daftarkan ke PATH."""
    kandidat = set()
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

# ==================== PENGATURAN ====================
BASE_DIR   = Path(__file__).resolve().parent
WAV_DIR    = BASE_DIR / "wav"
OUT_DIR    = BASE_DIR / "hasil"          # hasil disimpan di sini, struktur subfolder sama
SUBFOLDER  = ["full"]                    # ["part"], ["full"], atau ["part", "full"]
MODEL_SIZE = "large-v3-turbo"
BAHASA     = "id"
TIMPA      = False                       # False = lewati file yang hasilnya sudah ada
# ====================================================

if ada_cublas:
    device, compute_type = "cuda", "float16"
else:
    print("PERINGATAN: cublas64_12.dll tidak ditemukan, memakai CPU.\n")
    device, compute_type = "cpu", "int8"


def buat_model(dev, ctype):
    print(f"Memuat model ({MODEL_SIZE}) di {dev}...")
    m = WhisperModel(MODEL_SIZE, device=dev, compute_type=ctype)
    print("Model berhasil dimuat!\n")
    return m


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transkripsi(model, wav_path, out_txt, out_srt):
    segments, info = model.transcribe(
        str(wav_path),
        language=BAHASA,
        beam_size=5,
        vad_filter=True,
        log_progress=True,
    )
    print(f"Durasi audio: {info.duration:.2f} detik")

    with open(out_txt, "w", encoding="utf-8") as f_txt, \
         open(out_srt, "w", encoding="utf-8") as f_srt:
        for i, seg in enumerate(segments, 1):
            text = seg.text.strip()
            print(f"[{seg.start:7.2f}s → {seg.end:7.2f}s] {text}")
            f_txt.write(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}\n")
            f_srt.write(f"{i}\n")
            f_srt.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f_srt.write(f"{text}\n\n")


def hapus_sebagian(*paths):
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


# ---------- Kumpulkan daftar file ----------
daftar = []
for sub in SUBFOLDER:
    folder = WAV_DIR / sub
    if not folder.is_dir():
        print(f"Folder tidak ditemukan, dilewati: {folder}")
        continue
    for wav in sorted(folder.glob("*.wav")):
        daftar.append((sub, wav))

if not daftar:
    sys.exit("Tidak ada file .wav yang ditemukan.")

print(f"Total file: {len(daftar)}\n")

model = buat_model(device, compute_type)

berhasil, dilewati, gagal = [], [], []
mulai_total = time.time()

for n, (sub, wav) in enumerate(daftar, 1):
    folder_txt = OUT_DIR / "txt" / sub
    folder_srt = OUT_DIR / "srt" / sub
    folder_txt.mkdir(parents=True, exist_ok=True)
    folder_srt.mkdir(parents=True, exist_ok=True)
    out_txt = folder_txt / (wav.stem + ".txt")
    out_srt = folder_srt / (wav.stem + ".srt")

    print("=" * 60)
    print(f"[{n}/{len(daftar)}] {sub}/{wav.name}")
    print("=" * 60)

    if not TIMPA and out_txt.exists() and out_srt.exists():
        print("Hasil sudah ada, dilewati.\n")
        dilewati.append(wav.name)
        continue

    mulai = time.time()
    try:
        try:
            transkripsi(model, wav, out_txt, out_srt)
        except RuntimeError as e:
            if device == "cuda":
                print(f"\nGPU gagal ({e}). Beralih ke CPU (int8) untuk sisa file...\n")
                del model
                device, compute_type = "cpu", "int8"
                model = buat_model(device, compute_type)
                transkripsi(model, wav, out_txt, out_srt)
            else:
                raise
        berhasil.append(wav.name)
        print(f"\nSelesai dalam {time.time() - mulai:.1f} detik\n")
    except KeyboardInterrupt:
        hapus_sebagian(out_txt, out_srt)   # buang file setengah jadi
        print("\nDibatalkan oleh pengguna. File yang belum selesai dihapus.")
        break
    except Exception as e:
        hapus_sebagian(out_txt, out_srt)
        print(f"\nGAGAL: {wav.name} -> {e}\n")
        gagal.append(wav.name)

print("=" * 60)
print(f"Ringkasan (total waktu {(time.time() - mulai_total) / 60:.1f} menit)")
print(f"  Berhasil : {len(berhasil)}")
print(f"  Dilewati : {len(dilewati)}")
print(f"  Gagal    : {len(gagal)}")
for nama in gagal:
    print(f"    - {nama}")
print(f"Hasil ada di: {OUT_DIR}")