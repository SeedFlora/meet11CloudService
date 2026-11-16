# Lab 11 — RAG dengan pgvector, chat, dan sitasi

<!-- lecture-materials:start -->

## Materi teori sebelum praktikum

- [Pertemuan 11: NLP dan RAG](slides/Teori_Pertemuan_11.pptx)
- [Pertemuan 12: Proyek Cloud Native](slides/Teori_Pertemuan_12.pptx)
- [Pertemuan 13: Presentasi Proyek](slides/Teori_Pertemuan_13.pptx)

Slide menghubungkan konsep, kasus kerja, bacaan/video resmi, dan langkah lab. Sesi 12–13 dipakai untuk menyatukan dan mempresentasikan satu proyek kelompok.

<!-- lecture-materials:end -->

**Kebijakan kelas:** Lab ini latihan formatif, tanpa tugas, nilai, atau penyerahan terpisah. Satu proyek besar dikerjakan oleh kelompok **3 orang**, dengan presentasi checkpoint minggu 7 (UTS) dan hasil akhir minggu 14 (UAS). Simpan hasil lab hanya bila berguna sebagai referensi atau bukti proses proyek. Baca [brief proyek kelompok](PROYEK_KELOMPOK.md). Bobot resmi tetap mengikuti RPS/LMS.

**Capaian:** memotong dokumen, membuat embedding, menyimpan dan mencari vector dengan pgvector/HNSW, memakai fungsi `match_documents`, menyusun jawaban berbasis sumber, menguji API dan chat UI, lalu membandingkan latensi serta biaya. Jalur utama berjalan lokal tanpa akun cloud atau API berbayar.

Slide praktikum sesi 11 juga memperlihatkan rancangan cloud dengan TypeScript, Supabase, Groq, dan Next.js. Untuk praktik kelas yang dapat dijalankan dari folder ini, gunakan kode Python `rag.py`, `app.py`, `benchmark.py`, dan `index.html` di bawah. Contoh TypeScript pada slide dipakai untuk membahas rancangan integrasi cloud opsional; file TypeScript tersebut tidak disertakan sebagai aplikasi siap jalan.

## Mulai praktikum lokal

Ikuti [modul mahasiswa langkah demi langkah](MODUL_MAHASISWA.md). Modul memakai **PowerShell, clone baru, satu perintah tiap langkah**, hasil yang diharapkan, dan screenshot contoh. Semua langkah utama berjalan lokal. Tidak ada laporan atau push terpisah yang wajib dikumpulkan.

## 2. Pahami pipeline

`rag.py ingest` membaca `.md`/`.txt` dalam `docs/` lalu mengganti indeks secara transaksional. Mode default `hash` membuat vector hashing 384 dimensi tanpa model download. Ini **baseline leksikal**, bukan semantic embedding produksi. Pemotong default memakai **500 token demo + 50 overlap**: token demo dihitung dengan regex Unicode (kata dan tanda baca), sehingga **tidak identik** dengan token model LLM. Output ingest menampilkan jumlah bagian, chunk size, dan overlap. Tambahkan dokumen sendiri yang aman untuk dibagikan, indeks ulang, lalu amati perubahan jawaban.

Untuk semantic search yang sungguh memakai model, instal `requirements-semantic.txt`, ubah `EMBED_MODE=semantic` di `.env`, lalu jalankan ingest lagi. Mode ini memakai `sentence-transformers/all-MiniLM-L6-v2`, yang menghasilkan 384 dimensi dan membatasi input sekitar 256 word pieces. Kode memakai tokenizer model dan menetapkan 254 token sebagai batas awal chunk, lalu memendekkan chunk lagi bila teks hasil decode/re-encode bertambah token, dengan overlap sampai 50 tanpa celah. Ini sengaja berbeda dari angka 500 agar teks tidak terpotong diam-diam. Model tersebut terutama untuk bahasa Inggris; evaluasi pertanyaan bahasa Indonesia secara kritis. Model dapat diganti lewat `EMBED_MODEL` selama keluarannya **384 dimensi** (skema `vector(384)` tidak perlu diubah; kode menolak model dengan dimensi lain). Untuk dokumen bahasa Indonesia coba `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: batas inputnya 128 token sehingga chunk otomatis menjadi 126 token + overlap 31. Uji 16 November 2026 (CPU, PyTorch wheel CPU): ingest 3 dokumen menghasilkan 25 chunk; pertanyaan "Bagaimana menghentikan container tanpa menghapus data?" mengambil tiga chunk `docker_operations.md`, sedangkan pertanyaan tentang service role key justru mengambil `cloud_models.md`. Jadi evaluasi retrieval (misalnya hit@3 pada daftar pertanyaan uji) tetap wajib sebelum memilih model. Indeks ulang setiap kali `EMBED_MODE` atau `EMBED_MODEL` berubah.

Python 3.10+ juga memenuhi minimum [Sentence Transformers 5.7.0](https://pypi.org/project/sentence-transformers/5.7.0/) yang diperiksa dalam rentang dependency semantik. Paket dan unduhan model opsional perlu dipasang terpisah; baseline hash tidak memerlukannya.

`initdb/01_rag.sql` membuat HNSW dengan `vector_cosine_ops`. Fungsi SQL `match_documents` melakukan `ORDER BY embedding <=> query_embedding ASC LIMIT ...`. Urutan jarak naik dan operator langsung memungkinkan planner memakai indeks ketika ukuran tabel membenarkannya. Pada tiga dokumen kecil, PostgreSQL mungkin memilih sequential scan karena lebih murah; itu bukan bukti indeks rusak. Periksa dengan `EXPLAIN` saat dataset lebih besar.

## 3. LLM dan evaluasi opsional

Tanpa `GROQ_API_KEY`, jawaban ekstraktif diambil dari kalimat dokumen dan tetap menyertakan sitasi. Jika akun Groq tersedia, isi key **hanya** dalam `.env` lokal dan pilih model aktif sesuai dokumentasi Groq; server akan mengirim top-K sumber ke Chat Completions. Endpoint OpenAI-compatible lain dapat dipakai lewat `LLM_BASE_URL`, `LLM_MODEL`, dan `LLM_API_KEY` (bila diisi, prioritasnya di atas Groq): misalnya Ollama lokal tanpa GPU dan tanpa key (`LLM_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_MODEL=hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M` setelah `ollama run` model itu), Hugging Face router `https://router.huggingface.co/v1` (akun free perlu membeli kredit), atau Gemini `https://generativelanguage.googleapis.com/v1beta/openai`. Field `mode` pada respons menjadi `llm`, `groq`, atau `extractive`. Jalur generik ini diuji dengan server mock OpenAI-compatible; `tests/challenge.py` tetap 10 PASS pada mode ekstraktif. Jangan kirim catatan pribadi atau secret ke API luar. Periksa harga, rate limit, dan model yang tersedia pada hari lab.

PowerShell:

```powershell
.\.venv\Scripts\python benchmark.py
```

Bash/WSL:

```bash
.venv/bin/python benchmark.py
```

Simpan hasil `benchmarks.jsonl` sebagai data evaluasi lokal (file diabaikan Git). Buat tabel perbandingan mode hash vs semantic, ekstraktif vs LLM, p50/p95 latensi beberapa pengulangan, kualitas sitasi, serta estimasi biaya request. Jelaskan bahwa chunk 500/50 tidak selalu cocok untuk semua embedding model.

## 4. Jalur Supabase/Vercel (opsional untuk capstone)

Fungsi SQL lokal menunjukkan bentuk RPC `match_documents`, tetapi **jangan menyalin tabel/fungsi ini langsung ke project Supabase publik**. Atur RLS/grants untuk dokumen menurut pemilik atau jadikan corpus benar-benar publik, dan gunakan `SECURITY INVOKER` agar otorisasi caller berlaku. Jika menghosting RAG API, gunakan database managed yang dapat dijangkau server, simpan credential/LLM key di secret server, lalu panggil API itu dari Route Handler Next.js di Vercel. `localhost:5433` tidak dapat dijangkau dari deployment Vercel. Monitor endpoint RAG bersama aplikasi. Untuk kelas tanpa akun, demo lokal + push source ke GitHub sudah cukup untuk mencoba seluruh pipeline.

**Bukti:** jumlah chunk dan overlap, hasil top-K beserta jarak, satu jawaban dengan sitasi, percakapan dua giliran, benchmark latensi, dan diagram aliran pertanyaan → embedding → pgvector → sumber → jawaban. **Git opsional:** push source/SQL/dokumen contoh; jangan push `.env`, password, token, cache model, atau benchmark yang memuat dokumen pribadi.

Rujukan: [pgvector index usage](https://github.com/pgvector/pgvector#troubleshooting), [Supabase HNSW](https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Groq API](https://console.groq.com/docs/api-reference).

`benchmark.py` mengulang tiga pertanyaan lima kali secara default (15 sampel) dan mencetak p50 median serta p95 nearest-rank. Setelah Docker, ingest, dan Uvicorn hidup, jalankan PowerShell `.\.venv\Scripts\python -B tests\challenge.py` atau Bash `.venv/bin/python -B tests/challenge.py`; hasil uji **10 PASS, 0 FAIL**.

Panduan: [modul mahasiswa dan kunci](MODUL_MAHASISWA.md), [panduan Git](PANDUAN_GIT.md). Screenshot Docker Desktop, web, API, dan perintah berada di `screenshots/`.
