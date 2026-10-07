# Lab 11 — RAG dengan pgvector, chat, dan sitasi

**Capaian:** memotong dokumen, membuat embedding, menyimpan dan mencari vector dengan pgvector/HNSW, memakai fungsi `match_documents`, menyusun jawaban berbasis sumber, menguji API dan chat UI, lalu membandingkan latensi serta biaya. Jalur utama berjalan lokal tanpa akun cloud atau API berbayar.

Slide praktikum sesi 11 juga memperlihatkan rancangan cloud dengan TypeScript, Supabase, Groq, dan Next.js. Untuk praktik kelas yang dapat dijalankan dari folder ini, gunakan kode Python `rag.py`, `app.py`, `benchmark.py`, dan `index.html` di bawah. Contoh TypeScript pada slide dipakai untuk membahas rancangan integrasi cloud opsional; file TypeScript tersebut tidak disertakan sebagai aplikasi siap jalan.

## 1. Siapkan database dan Python

Dari root repo Lab 11:

PowerShell:

```powershell
Copy-Item .\secrets\db_password.txt.example .\secrets\db_password.txt
Copy-Item .\.env.example .\.env
docker compose up -d --wait
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python rag.py ingest
.\.venv\Scripts\python rag.py ask "Mengapa RLS penting untuk Supabase?"
.\.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8011
```

Bash/WSL:

```bash
cp secrets/db_password.txt.example secrets/db_password.txt
cp .env.example .env
docker compose up -d --wait
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python rag.py ingest
.venv/bin/python rag.py ask "Mengapa RLS penting untuk Supabase?"
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8011
```

`/health` memeriksa proses API; `/ready` juga memeriksa database, jumlah chunk, dan mode embedding (503 bila belum siap). Buka `http://127.0.0.1:8011`. Coba percakapan dan lihat `[S1]` serta nama file/bagian pada sumber. `http://127.0.0.1:8011/docs` membuka dokumentasi API. Uji POST dari PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8011/api/ask -Method Post -ContentType application/json -Body '{"question":"Apa perbedaan image dan container?","top_k":3}'
```

Atau bash:

```bash
curl -fsS http://127.0.0.1:8011/api/ask -H 'Content-Type: application/json' -d '{"question":"Apa perbedaan image dan container?","top_k":3}'
```

## 2. Pahami pipeline

`rag.py ingest` membaca `.md`/`.txt` dalam `docs/` lalu mengganti indeks secara transaksional. Mode default `hash` membuat vector hashing 384 dimensi tanpa model download. Ini **baseline leksikal**, bukan semantic embedding produksi. Pemotong default memakai **500 token demo + 50 overlap**: token demo dihitung dengan regex Unicode (kata dan tanda baca), sehingga **tidak identik** dengan token model LLM. Output ingest menampilkan jumlah bagian, chunk size, dan overlap. Tambahkan dokumen sendiri yang aman untuk dibagikan, indeks ulang, lalu amati perubahan jawaban.

Untuk semantic search yang sungguh memakai model, instal `requirements-semantic.txt`, ubah `EMBED_MODE=semantic` di `.env`, lalu jalankan ingest lagi. Mode ini memakai `sentence-transformers/all-MiniLM-L6-v2`, yang menghasilkan 384 dimensi dan membatasi input sekitar 256 word pieces. Kode memakai tokenizer model dan menetapkan 254 token sebagai batas awal chunk, lalu memendekkan chunk lagi bila teks hasil decode/re-encode bertambah token, dengan overlap sampai 50 tanpa celah. Ini sengaja berbeda dari angka 500 agar teks tidak terpotong diam-diam. Model tersebut terutama untuk bahasa Inggris; evaluasi pertanyaan bahasa Indonesia secara kritis.

`initdb/01_rag.sql` membuat HNSW dengan `vector_cosine_ops`. Fungsi SQL `match_documents` melakukan `ORDER BY embedding <=> query_embedding ASC LIMIT ...`. Urutan jarak naik dan operator langsung memungkinkan planner memakai indeks ketika ukuran tabel membenarkannya. Pada tiga dokumen kecil, PostgreSQL mungkin memilih sequential scan karena lebih murah; itu bukan bukti indeks rusak. Periksa dengan `EXPLAIN` saat dataset lebih besar.

## 3. LLM dan evaluasi opsional

Tanpa `GROQ_API_KEY`, jawaban ekstraktif diambil dari kalimat dokumen dan tetap menyertakan sitasi. Jika akun Groq tersedia, isi key **hanya** dalam `.env` lokal dan pilih model aktif sesuai dokumentasi Groq; server akan mengirim top-K sumber ke Chat Completions. Jangan kirim catatan pribadi atau secret ke API luar. Periksa harga, rate limit, dan model yang tersedia pada hari lab.

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

Panduan: [modul mahasiswa dan kunci](MODUL_MAHASISWA.md), [panduan dosen](PANDUAN_DOSEN.md), [panduan Git](PANDUAN_GIT.md). Screenshot Docker Desktop, web, API, dan perintah berada di `screenshots/`.

