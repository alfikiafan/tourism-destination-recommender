# Indonesia Tourism Recommendation

**Indonesia Tourism Recommendation** adalah sebuah sistem rekomendasi yang dirancang untuk membantu wisatawan menemukan destinasi wisata di Indonesia yang sesuai dengan preferensi mereka. Sistem ini mengimplementasikan dua pendekatan utama yaitu **Content-based Filtering** dan **Collaborative Filtering** untuk memberikan rekomendasi yang relevan dan personal.

Dengan meningkatnya jumlah wisatawan, sistem rekomendasi ini diharapkan dapat meningkatkan pengalaman pengguna dalam menemukan destinasi wisata yang menarik dan sesuai dengan keinginan mereka, sekaligus mendukung pengembangan pariwisata di berbagai daerah di Indonesia.

Untuk laporan lengkap, silakan baca dokumen [Laporan Proyek](report.md).

## Daftar Isi

- [Fitur](#fitur)
- [Dataset](#dataset)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Model](#model)
- [Evaluasi](#evaluasi)

## Fitur

- **Content-based Filtering**: Memberikan rekomendasi berdasarkan kesamaan konten deskripsi dan kategori destinasi wisata.
- **Collaborative Filtering**: Menggunakan data rating pengguna untuk memberikan rekomendasi yang lebih personal.
- **Evaluasi Model**: Menggunakan metrik Precision@10, RMSE, dan MAE untuk menilai performa model.
- **Visualisasi Data**: Menampilkan visualisasi distribusi rating, jumlah destinasi per kategori, dan jumlah rating per destinasi.

## Dataset

Proyek ini menggunakan dua dataset utama:

1. **tourism_with_id.csv**: Berisi informasi detail tentang destinasi wisata di Indonesia.
2. **tourism_rating.csv**: Berisi data rating yang diberikan oleh pengguna terhadap destinasi wisata tertentu.

### Sumber Dataset

Dataset dapat diunduh melalui Kaggle:  
[Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

### Deskripsi Atribut

#### tourism_with_id.csv

| No. | Kode Atribut | Deskripsi                                      | Tipe Data |
|-----|--------------|------------------------------------------------|-----------|
| 1   | Place_Id     | Identifikasi unik untuk setiap destinasi wisata.| Integer   |
| 2   | Place_Name   | Nama destinasi wisata.                         | String    |
| 3   | Description  | Deskripsi singkat tentang destinasi.           | String    |
| 4   | Category     | Kategori destinasi (misalnya, Budaya, Alam).   | String    |
| 5   | City         | Kota tempat destinasi berada.                  | String    |
| 6   | Price        | Harga tiket masuk.                             | Integer   |
| 7   | Rating       | Rating rata-rata dari pengguna.                | Float     |
| 8   | Time_Minutes | Waktu yang dibutuhkan untuk mengunjungi.       | Float     |
| 9   | Coordinate   | Koordinat geografis.                           | String    |
| 10  | Lat          | Latitude lokasi.                               | Float     |
| 11  | Long         | Longitude lokasi.                              | Float     |
| 12  | Unnamed: 11  | Kolom kosong.                                  | Float     |
| 13  | Unnamed: 12  | Kolom kosong.                                  | Integer   |

#### tourism_rating.csv

| No. | Kode Atribut  | Deskripsi                                                   | Tipe Data |
|-----|---------------|-------------------------------------------------------------|-----------|
| 1   | User_Id       | Identifikasi unik untuk setiap pengguna                    | Integer   |
| 2   | Place_Id      | Identifikasi unik untuk setiap destinasi wisata             | Integer   |
| 3   | Place_Ratings | Rating yang diberikan oleh pengguna kepada destinasi wisata  | Integer   |

## Instalasi

Untuk menjalankan proyek ini, Anda memerlukan beberapa dependensi. Berikut adalah langkah-langkah instalasinya:

1. **Clone Repository**

    ```bash
    git clone https://github.com/alfikiafan/indonesia-tourism-recommendation.git
    cd indonesia-tourism-recommendation
    ```

2. **Buat dan Aktifkan Virtual Environment**

    ```bash
    python -m venv env
    source env/bin/activate  # Untuk Linux/Mac
    env\Scripts\activate     # Untuk Windows
    ```

3. **Install Dependensi**

    ```bash
    pip install -r requirements.txt
    ```

4. **Mengatur API Kaggle**

    - Unduh `kaggle.json` dari akun Kaggle Anda.
    - Upload `kaggle.json` ke direktori kerja Anda.

    ```bash
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

## Penggunaan

Proyek ini disusun dalam format notebook Jupyter yang dapat dijalankan di Google Colab atau lingkungan lokal Anda. Berikut adalah langkah-langkah untuk menjalankannya:

1. **Mengunduh Dataset**

    Dataset dapat diunduh secara otomatis menggunakan script yang telah disediakan dalam notebook.

2. **Menjalankan Notebook**

    Buka `destination_recommender.ipynb` dan jalankan setiap sel secara berurutan untuk memproses data, membangun model, dan mengevaluasi performa.

3. **Rekomendasi Destinasi**

    Anda dapat memberikan nama destinasi wisata tertentu untuk mendapatkan rekomendasi dari model Content-based Filtering dan Collaborative Filtering.

## Model

### 1. Content-based Filtering

Menggunakan informasi konten seperti deskripsi dan kategori untuk memberikan rekomendasi. Proses meliputi:

- **Preprocessing Teks**: Membersihkan teks dengan lowercase, stemming, dan menghapus stopword.
- **Vektorisasi Teks**: Mengubah teks menjadi vektor numerik menggunakan TF-IDF Vectorizer.
- **Menghitung Similarity**: Menggunakan cosine similarity untuk mengukur kesamaan antar destinasi.

### 2. Collaborative Filtering

Memanfaatkan data rating pengguna untuk memberikan rekomendasi. Proses meliputi:

- **Encoding Data**: Mengubah `User_Id` dan `Place_Id` menjadi indeks numerik.
- **Normalisasi Rating**: Menormalisasi rating agar berada di antara 0 dan 1.
- **Membangun Model Neural Network**: Menggunakan embedding layers untuk pengguna dan tempat.
- **Melatih Model**: Melatih model menggunakan data latih dan validasi.
- **Fungsi Rekomendasi**: Menghasilkan rekomendasi berdasarkan prediksi rating tertinggi.

## Evaluasi

### Metrik yang Digunakan

- **Precision**: Mengukur proporsi rekomendasi yang relevan di antara 10 rekomendasi teratas.
- **Root Mean Squared Error (RMSE)**: Mengukur rata-rata perbedaan antara nilai prediksi dan nilai aktual.
- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual.

### Hasil Evaluasi

- **Content-based Filtering**  
  **Precision**: 100.00% (bervariasi untuk setiap destinasi)
  
- **Collaborative Filtering**
  - **RMSE**: 0.3583
  - **MAE**: 0.3105

### Interpretasi Hasil

- **Content-based Filtering** menunjukkan tingkat presisi yang baik dalam memberikan rekomendasi yang relevan berdasarkan konten deskripsi dan kategori destinasi.
- **Collaborative Filtering** memiliki performa yang moderat dengan RMSE dan MAE yang menunjukkan akurasi prediksi yang baik, meskipun masih ada ruang untuk peningkatan.

---

*Terima kasih telah menggunakan Indonesia Tourism Recommendation! Semoga membantu dalam menemukan destinasi wisata impian Anda di Indonesia.*
