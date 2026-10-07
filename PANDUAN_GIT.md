# Git untuk repo Lab 11

Repo materi dosen: https://github.com/SeedFlora/meet11CloudService. Gunakan **Use this template** di GitHub untuk membuat repo milik sendiri (atau fork jika tombol template belum tersedia). Buka Codespaces dari repo milik sendiri bila memakai browser. Di komputer lokal, clone **URL repo milik sendiri**, lalu buka terminal pada root clone. `git remote -v` harus menunjuk repo Anda sebelum push.

```bash
git remote -v
git status --short
git add .
git diff --cached --name-only
git diff --cached --check
git commit -m "lab11: praktik dan laporan"
git push -u origin main
```

Jika branch repo Anda bukan `main`, jalankan `git branch --show-current` lalu gunakan nama branch itu. Jika Git meminta identitas pertama kali, set `git config user.name "Nama Anda"` dan `git config user.email "email@contoh.com"` di repo Anda. Jangan memasukkan token/password dalam URL remote. Periksa daftar staged; file `.env`, password lokal, `.venv`, `node_modules`, `.next`, dan hasil operasi yang sensitif tidak boleh ikut. Laporan pribadi dapat disimpan di `hasil/lab11.md` dengan screenshot milik Anda di `hasil/bukti/`.
