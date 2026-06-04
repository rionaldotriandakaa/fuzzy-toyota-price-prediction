# Kelompok 6 - Dasar Kecerdasan Artifisial (DKA)
## S1 Informatika - Telkom University

### 🚗 Penerapan Sistem Inferensi Fuzzy Sugeno dan Mamdani untuk Mengestimasi Harga Jual Mobil Bekas Toyota Berdasarkan Karakteristik Spesifikasi Kendaraan

Repositori ini dibuat untuk memenuhi projek Tugas Besar pada mata kuliah **Dasar Kecerdasan Artifisial (DKA)** tahun ajaran 2025/2026. Projek ini berfokus pada perancangan, implementasi, serta analisis komparatif antara **Sistem Inferensi Fuzzy (FIS)** metode **Mamdani** dan **Sugeno** dalam melakukan estimasi harga jual mobil bekas khusus untuk merk Toyota.

---

### 📋 Deskripsi Projek

Menentukan harga wajar bagi kendaraan bekas merupakan tantangan nyata yang dihadapi oleh diler maupun calon pembeli akibat banyaknya faktor spesifikasi fisik dan historis kendaraan yang bervariasi. Melalui pendekatan *Fuzzy Logic*, sistem ini mampu memberikan estimasi harga berdasarkan penalaran logika manusia (*linguistic rules*) yang terukur, objektif, dan mudah diinterpretasikan, alih-alih mengandalkan intuisi subjektif pasar semata.

Sistem ini memproses **5 Variabel Input** numerik yang diekstrak dari karakteristik riil kendaraan untuk menghasilkan **1 Variabel Output** berupa estimasi harga prediktif. Projek ini mengevaluasi dan membandingkan dua metode utama:
1. **Fuzzy Mamdani:** Menggunakan fungsi keanggotaan pada konsekuen (output) dan melakukan defuzzifikasi dengan metode *Centroid* (titik berat).
2. **Fuzzy Sugeno (Orde-1):** Menggunakan fungsi linear pada bagian konsekuen yang dikalkulasi secara efisien menggunakan rumus rata-rata terbobot (*Weighed Average*).

Performa dari kedua metode ini diuji secara komparatif untuk mengukur tingkat akurasi (menggunakan metrik eror seperti MAE/RMSE) serta efisiensi waktu komputasi (*inference running time*) saat mengolah ribuan baris data.

---

### 📊 Spesifikasi Dataset & Kesesuaian Aturan

* **Link Sumber Dataset:** [Kaggle - Toyota Used Car Listing](https://www.kaggle.com/datasets/mysarahmadbhat/toyota-used-car-listing)
* **Sifat Data:** Data Nyata (Hasil *scraping* daftar penjualan asli mobil bekas Toyota).
* **Jumlah Baris:** 6.738 baris data (Memenuhi syarat minimal tugas besar > 5.000 baris).
* **Variabel Input (5 Fitur):**
  1. `year` (Tahun): Tahun pembuatan/perakitan mobil (merepresentasikan usia pakai kronologis).
  2. `mileage` (Jarak Tempuh): Akumulasi jarak yang telah ditempuh kendaraan dalam satuan mil (merepresentasikan tingkat keausan mesin).
  3. `tax` (Pajak): Besaran beban pajak tahunan resmi kendaraan (£).
  4. `mpg` (*Miles per Gallon*): Tingkat efisiensi konsumsi bahan bakar operasional.
  5. `engineSize` (Kapasitas Mesin): Ukuran volume silinder total pada komponen mesin (Liter).
* **Variabel Output (1 Target):**
  * `price` (Harga Jual): Estimasi harga jual pasar mobil bekas (£).

---

### 🛠️ Alur Metodologi Eksekusi (*Pipeline*)

1. **Exploratory Data Analysis (EDA) & Preprocessing:** Pembersihan data kosong (*missing values*) dan penanganan data jika terdapat anomali, dilanjutkan analisis statistik deskriptif untuk memetakan batas semesta pembicaraan (*universe of discourse*) fuzzy.
2. **Fuzzifikasi:** Pembentukan fungsi keanggotaan (Kurva Segitiga & Trapesium) untuk mengubah variabel input tegas (*crisp input*) menjadi nilai linguistik (misal: Jarak tempuh *Rendah, Sedang, Tinggi*).
3. **Rule Base Construction:** Penyusunan kombinasi aturan logika *IF-THEN* berdasarkan penalaran logika korelasi data spesifikasi kendaraan.
4. **Inference Engine & Defuzzifikasi:** Eksekusi mesin inferensi Mamdani (MIN-MAX & Centroid) serta Sugeno Orde-1 (*Weighted Average*).
5. **Evaluasi Komparatif:** Pengujian menyeluruh pada 5.000+ baris data untuk membandingkan tingkat eror prediksi dan kecepatan *running time* program.

---

### 👥 Anggota Kelompok 6 (IF-48-06)

* **RIONALDO TRIANDAKA** - NIM: 103012430001
* **Rahmatul Akbar Alim** - NIM: 103012400172
* **Muhammad Rafiul Izzah** - NIM: 103012430004

---

### ⚙️ Persyaratan Sistem (Prerequisites)

Untuk menjalankan kodingan notebook projek ini, pastikan Anda telah menginstal pustaka Python berikut di lingkungan lokal Anda:

```bash
pip install numpy pandas matplotlib scikit-fuzzy
