"""Local web UI for faster-whisper. Run with the project's Python environment."""

from __future__ import annotations

import glob
import json
import mimetypes
import os
import site
import sys
import threading
import time
import traceback
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "hasil" / "uploads"
OUTPUT_DIR = BASE_DIR / "hasil" / "web"
MAX_UPLOAD = 2 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
MODEL_CACHE: dict = {}


def register_cuda_dlls() -> list[str]:
    found = []
    roots = list(sys.path)
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    for root in set(roots):
        for lib in ("cublas", "cudnn"):
            directory = os.path.join(root, "nvidia", lib, "bin")
            if os.path.isdir(directory):
                try:
                    os.add_dll_directory(directory)
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
                found.append(directory)
    return found


CUDA_DLLS = register_cuda_dlls()
from faster_whisper import WhisperModel


def format_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d},{ms:03d}"


def run_transcription(job_id: str, audio: Path, options: dict) -> None:
    job = JOBS[job_id]
    model = None
    started = time.time()
    try:
        requested_device = options["device"]
        device = requested_device
        compute = options["compute_type"]
        if device == "auto":
            has_cublas = any(glob.glob(os.path.join(p, "cublas64_12.dll")) for p in CUDA_DLLS)
            device, compute = ("cuda", "float16") if has_cublas else ("cpu", "int8")

        with MODEL_LOCK:
            job.update(status="running", message=f"Memuat model {options['model']} di {device}…", device=device)
            try:
                cache_key = (options["model"], device, compute)
                if MODEL_CACHE.get("key") != cache_key:
                    MODEL_CACHE["model"] = WhisperModel(options["model"], device=device, compute_type=compute)
                    MODEL_CACHE["key"] = cache_key
                model = MODEL_CACHE["model"]
                try:
                    segments, info = model.transcribe(
                        str(audio), language=options["language"] or None,
                        beam_size=options["beam_size"], vad_filter=options["vad_filter"],
                    )
                    job["duration"] = info.duration
                    job["language_detected"] = info.language
                    job["progress"] = 1
                    txt_lines, srt_lines, rows = [], [], []
                    for index, seg in enumerate(segments, 1):
                        text = seg.text.strip()
                        txt_lines.append(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}")
                        srt_lines.extend([str(index), f"{format_time(seg.start)} --> {format_time(seg.end)}", text, ""])
                        rows.append({"start": seg.start, "end": seg.end, "text": text})
                        job["segments"] = rows
                        job["progress"] = min(99, int(seg.end / info.duration * 100)) if info.duration else 1
                        job["message"] = f"Memproses segmen {index}…"
                    out_txt = OUTPUT_DIR / f"{job_id}.txt"
                    out_srt = OUTPUT_DIR / f"{job_id}.srt"
                    out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
                    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
                    job.update(status="completed", message="Transkripsi selesai", elapsed=time.time() - started,
                               progress=100, txt=str(out_txt), srt=str(out_srt))
                except RuntimeError:
                    if device != "cuda":
                        raise
                    job["message"] = "GPU bermasalah; mencoba ulang dengan CPU…"
                    del model
                    model = WhisperModel(options["model"], device="cpu", compute_type="int8")
                    MODEL_CACHE.update(key=(options["model"], "cpu", "int8"), model=model)
                    segments, info = model.transcribe(str(audio), language=options["language"] or None,
                                                      beam_size=options["beam_size"], vad_filter=options["vad_filter"])
                    rows, txt_lines, srt_lines = [], [], []
                    job["duration"] = info.duration
                    job["language_detected"] = info.language
                    job["progress"] = 1
                    for index, seg in enumerate(segments, 1):
                        text = seg.text.strip()
                        rows.append({"start": seg.start, "end": seg.end, "text": text})
                        txt_lines.append(f"[{seg.start:.2f}s → {seg.end:.2f}s] {text}")
                        srt_lines.extend([str(index), f"{format_time(seg.start)} --> {format_time(seg.end)}", text, ""])
                        job["segments"] = rows
                        job["progress"] = min(99, int(seg.end / info.duration * 100)) if info.duration else 1
                    out_txt, out_srt = OUTPUT_DIR / f"{job_id}.txt", OUTPUT_DIR / f"{job_id}.srt"
                    out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
                    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")
                    job.update(status="completed", message="Selesai menggunakan CPU", device="cpu",
                               elapsed=time.time() - started, progress=100, txt=str(out_txt), srt=str(out_srt))
            except Exception:
                raise
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=str(exc), elapsed=time.time() - started)
    finally:
        try:
            audio.unlink(missing_ok=True)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    server_version = "WhisperLocal/1.0"

    def send_json(self, value: dict, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            data = (BASE_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif "/download/" in path:
            bits = path.strip("/").split("/")
            job = JOBS.get(bits[2]) if len(bits) == 5 else None
            ext = bits[-1]
            if not job or ext not in ("txt", "srt") or job.get("status") != "completed":
                self.send_json({"error": "Hasil tidak tersedia"}, 404)
                return
            file = Path(job[ext])
            data = file.read_bytes()
            download_name = f"{Path(job['filename']).stem}.{ext}"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename=transcript.{ext}; filename*=UTF-8''{quote(download_name)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                self.send_json({"error": "Pekerjaan tidak ditemukan"}, 404)
            else:
                result = {k: v for k, v in job.items() if k not in ("txt", "srt")}
                result["downloads"] = {ext: f"/api/jobs/{job_id}/download/{ext}" for ext in ("txt", "srt")} if job["status"] == "completed" else {}
                self.send_json(result)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/transcribe":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                self.send_json({"error": "Ukuran file kosong atau melewati batas 2 GB"}, 413)
                return
            raw = self.rfile.read(length)
            message = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
            )
            fields = {}
            uploads = []
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "audio":
                    filename = Path(unquote(part.get_filename() or "audio.wav")).name
                    uploads.append((filename, part.get_payload(decode=True)))
                elif name:
                    fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
            uploads = [(name, data) for name, data in uploads if name.lower().endswith(".wav") and data]
            if not uploads:
                self.send_json({"error": "Pilih minimal satu file WAV yang valid"}, 400)
                return
            options = {
                "model": fields.get("model", "large-v3-turbo"),
                "language": fields.get("language", "id"),
                "device": fields.get("device", "auto"),
                "compute_type": fields.get("compute_type", "float16"),
                "beam_size": max(1, min(10, int(fields.get("beam_size", "5")))),
                "vad_filter": fields.get("vad_filter", "true") == "true",
            }
            if options["model"] not in ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"):
                self.send_json({"error": "Pilihan model tidak valid"}, 400)
                return
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            results = []
            for filename, audio_data in uploads:
                job_id = uuid.uuid4().hex
                audio = UPLOAD_DIR / f"{job_id}.wav"
                audio.write_bytes(audio_data)
                JOBS[job_id] = {"id": job_id, "filename": filename, "status": "queued", "message": "Menunggu proses…", "segments": [], "progress": 0, "created": time.time()}
                threading.Thread(target=run_transcription, args=(job_id, audio, options), daemon=True).start()
                results.append({"id": job_id, "filename": filename})
            self.send_json({"jobs": results}, 202)
        except (ValueError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    host = "127.0.0.1"
    server = None
    for port in (8000, 8765, 8766, 8888, 9000):
        try:
            server = ThreadingHTTPServer((host, port), Handler)
            break
        except OSError as exc:
            print(f"Port {port} tidak bisa digunakan: {exc}")
    if server is None:
        raise SystemExit("Tidak dapat membuka port lokal. Coba tutup aplikasi lain atau periksa pengaturan firewall.")
    print(f"Whisper UI tersedia di http://{host}:{server.server_address[1]}")
    print("Tekan Ctrl+C untuk menghentikan server.")
    server.serve_forever()
