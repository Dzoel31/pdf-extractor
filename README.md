# PDF Extractor

Aplikasi Streamlit untuk ekstraksi PDF, OCR, dan transkripsi audio/video menggunakan Whisper `large-v3`.

## Menjalankan dengan Docker

CPU adalah varian standar dan cocok untuk laptop tanpa NVIDIA GPU:

```bash
docker pull dzuladj/pdf-extractor:latest
docker run --rm -p 8501:8501 \
  -v pdf-extractor-models:/app/app/models \
  dzuladj/pdf-extractor:latest
```

Buka <http://localhost:8501>. Model Whisper `large-v3` diunduh ketika transkripsi pertama dijalankan, bukan dimasukkan ke image. Volume menyimpan model agar tidak perlu diunduh ulang. Container juga menyiapkan model Docling saat pertama kali berjalan.

Untuk NVIDIA GPU di mesin Linux dengan NVIDIA Container Toolkit, gunakan image `latest-gpu` dan port 8502:

```bash
docker pull dzuladj/pdf-extractor:latest-gpu
docker run --rm --gpus all -p 8502:8501 \
  -v pdf-extractor-models:/app/app/models \
  dzuladj/pdf-extractor:latest-gpu
```

## Menjalankan dari source

Perlu Python 3.13, [uv](https://docs.astral.sh/uv/), FFmpeg, dan Tesseract OCR. FFmpeg dipakai Whisper untuk membaca audio/video; Tesseract dipakai OCR. Contoh pemasangan tool sistem:

- Windows: pasang FFmpeg dari [ffmpeg.org](https://ffmpeg.org/download.html) dan Tesseract dari [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki); pastikan executable tersedia di `PATH`.
- macOS: `brew install ffmpeg tesseract tesseract-lang`.
- Debian/Ubuntu: `sudo apt-get install ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind`.

```bash
git clone https://github.com/Dzoel31/pdf-extractor.git
cd pdf-extractor
uv python install 3.13
uv sync --locked --no-dev --extra cpu --extra windows
```

Salin `.env.example` menjadi `.env`, kemudian isi hanya kredensial untuk fitur opsional yang digunakan (rinciannya di bawah). Jalankan aplikasi:

```bash
uv run --no-sync streamlit run dashboard.py
```

Buka <http://localhost:8501>.

## Konfigurasi `.env`

Salin template:

```bash
cp .env.example .env
```

Di PowerShell:

```powershell
Copy-Item .env.example .env
```

Azure Document Intelligence tidak wajib. Biarkan kedua nilai ini kosong jika tidak memilih metode **Azure Doc Intelligence** di aplikasi. Isi endpoint dan key Azure hanya ketika metode itu dipilih:

```dotenv
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
```

Perapian hasil transkripsi melalui Gemini juga opsional. Isi `GOOGLE_API_KEY` (atau `GEMINI_API_KEY`) hanya jika fitur itu ingin digunakan. Tanpa key, Whisper tetap menghasilkan transkrip dan langkah Gemini dilewati.

Jangan commit `.env` atau memasukkan key ke image. Berkas `.env` sudah diabaikan oleh Git.

## Build image lokal

Build CPU untuk pengembangan lokal:

```bash
docker build --build-arg TORCH_VARIANT=cpu -t pdf-extractor:cpu-dev .
```

Compose:

```bash
docker compose up --build pdf-extractor
```

Build varian GPU tersedia untuk host Linux NVIDIA; tidak perlu menjalankan inference GPU untuk membangun image:

```bash
docker compose --profile gpu up --build pdf-extractor-gpu
```

## Release dan Docker Hub

Setelah CI lulus di `main`, Semantic Release membuat GitHub Release sesuai Conventional Commits. Workflow Docker Release menerbitkan dua image ke Docker Hub:

- `dzuladj/pdf-extractor:latest` dan `<versi>-cpu` untuk CPU.
- `dzuladj/pdf-extractor:latest-gpu` dan `<versi>-gpu` untuk CUDA GPU.

GitHub Actions memerlukan repository secrets `DOCKER_USERNAME`, `DOCKER_PUSH` (Docker Hub access token), dan `RELEASE_TOKEN` (GitHub token yang dapat membuat release dan mendorong tag/commit release). CI membangun kedua varian; runner CI tidak menjalankan inference GPU karena tidak memiliki GPU fisik.
