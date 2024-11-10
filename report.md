# Laporan Proyek Machine Learning

## Project Overview

### Latar Belakang

Indonesia memiliki kekayaan alam dan budaya yang melimpah, menjadikannya sebagai salah satu destinasi wisata utama di dunia. Berdasarkan data dari Badan Pusat Statistik (BPS), pada tahun 2019, jumlah wisatawan mancanegara yang berkunjung ke Indonesia mencapai 16,11 juta orang, meningkat dari tahun-tahun sebelumnya [[1](https://www.bps.go.id/id/pressrelease/2020/02/03/1711/jumlah-kunjungan-wisman-ke-indonesia-desember-2019-mencapai-1-38-juta-kunjungan-.html)]. Namun, pada tahun 2020 hingga 2021, terjadi penurunan tajam akibat pandemi COVID-19, dengan kunjungan menurun sekitar 75% dibandingkan periode sebelumnya [[2](https://www.unwto.org/impact-assessment-of-the-covid-19-outbreak-on-international-tourism)]. Meskipun demikian, pariwisata domestik tetap menjadi andalan bagi perekonomian nasional, dan pemulihan pariwisata kini menjadi prioritas utama bagi pemerintah Indonesia.

Dalam upaya memulihkan dan meningkatkan kembali sektor pariwisata, diperlukan strategi yang efektif untuk menarik minat wisatawan. Salah satu tantangan yang dihadapi wisatawan adalah kesulitan dalam memilih destinasi yang sesuai dengan preferensi mereka, mengingat banyaknya pilihan yang tersedia. Tanpa panduan yang tepat, wisatawan mungkin mengalami kebingungan, yang dapat mengurangi kepuasan mereka selama berwisata [[3](https://senti.ft.ugm.ac.id/wp-content/uploads/sites/454/2024/10/Pengembangan-Sistem-Rekomendasi-Rencana-Perjalanan-Literature-Review.pdf)]. Oleh karena itu, pengembangan sistem rekomendasi destinasi wisata menjadi penting untuk membantu wisatawan dalam menentukan pilihan yang sesuai dengan minat dan kebutuhan mereka.

Penelitian terkait sistem rekomendasi dalam bidang pariwisata telah banyak dilakukan. Misalnya, studi oleh Naatonis (2019) mengembangkan sistem rekomendasi destinasi wisata di Kota Kupang menggunakan metode Weighted Product untuk membantu wisatawan memilih destinasi berdasarkan kriteria biaya, fasilitas, dan ulasan pengunjung [[4](https://publikasi.uyelindo.ac.id/index.php/hoaq/article/view/15)]. Selain itu, juga terdapat penelitian oleh Murzani dan Sari (2023) mengimplementasikan metode collaborative filtering untuk merekomendasikan destinasi wisata di Aceh, menghasilkan 7 rekomendasi teratas bagi pengguna [[5](https://jurnal.usk.ac.id/kitektro/article/view/36168)]

Di sisi lain, collaborative filtering yang memanfaatkan data rating dari pengguna dapat menghasilkan rekomendasi yang lebih akurat dengan mengenali pola preferensi pengguna. Namun, metode ini memiliki keterbatasan dalam cold-start problem, di mana sistem mengalami kesulitan memberikan rekomendasi untuk pengguna atau item baru tanpa riwayat interaksi [[6](https://link.springer.com/book/10.1007/978-1-4899-7637-6)].

Proyek ini akan menggunakan kombinasi metode TF-IDF (Term Frequency-Inverse Document Frequency) [[7](https://www.emerald.com/insight/content/doi/10.1108/00220410410560582/full/html)], content-based filtering dengan cosine similarity, serta collaborative filtering dengan neural network untuk menghasilkan rekomendasi yang optimal. TF-IDF akan digunakan untuk mengubah deskripsi destinasi menjadi vektor numerik, memungkinkan sistem untuk menghitung kesamaan antar destinasi menggunakan cosine similarity. Dalam hal ini, cosine similarity efektif untuk mengukur relevansi antar destinasi berdasarkan fitur teks, seperti deskripsi dan kategori [[6](https://link.springer.com/book/10.1007/978-1-4899-7637-6)]. Sementara itu, collaborative filtering berbasis neural network dapat mengatasi masalah cold-start melalui pemetaan preferensi pengguna ke dalam embedding yang terus diperbarui [[8](https://ieeexplore.ieee.org/document/9338389)].

Untuk mengatasi mendukung pemulihan sektor pariwisata, sistem rekomendasi ini akan membantu wisatawan dalam menemukan destinasi wisata yang sesuai dengan preferensi mereka. Dengan pendekatan ini, diharapkan sistem rekomendasi mampu memberikan hasil yang relevan dan akurat meskipun terdapat variasi preferensi pengguna yang luas.

## Business Understanding

### Problem Statements
1. **Bagaimana memberikan rekomendasi destinasi wisata yang relevan berdasarkan konten deskripsi dan kategori?**  
   Destinasi wisata yang dipilih harus sesuai dengan preferensi pengguna, berdasarkan informasi yang ada pada deskripsi dan kategori destinasi. Permasalahan ini membutuhkan solusi yang mampu memanfaatkan informasi spesifik tentang tempat, seperti kategori destinasi (budaya, alam, dll.) serta deskripsinya.

2. **Bagaimana memanfaatkan data rating pengguna untuk meningkatkan akurasi rekomendasi?**  
   Sistem rekomendasi perlu mempertimbangkan rating yang diberikan oleh pengguna sebelumnya untuk memperbaiki kualitas rekomendasi yang diberikan.

3. **Bagaimana mengukur relevansi rekomendasi untuk memastikan sistem memberikan hasil yang tepat?**  
   Sistem rekomendasi perlu memberikan rekomendasi yang sesuai dan relevan dengan kebutuhan pengguna, sehingga diperlukan metrik evaluasi yang dapat memantau dan mengukur kinerja sistem dalam memberikan hasil yang sesuai.

### Goals
1. **Membangun sistem rekomendasi berbasis konten yang efektif menggunakan deskripsi dan kategori destinasi wisata.**  
   Informasi yang tersedia pada deskripsi dan kategori dapat dimanfaatkan untuk menghasilkan rekomendasi yang sesuai dengan minat pengguna yang telah mengunjungi lokasi wisata tertentu.

2. **Membangun sistem rekomendasi destinasi wisata berdasarkan pola rating pengguna.**  
   Sistem rekomendasi dapat dibangun dengan menggunakan pendekatan yang mempertimbangkan interaksi atau rating dari pengguna. Hal ini untuk memastikan bahwa rekomendasi mencerminkan pengalaman atau preferensi kolektif dari pengguna-pengguna lainnya.

3. **Mengukur performa sistem rekomendasi menggunakan metrik evaluasi yang sesuai.**  
   Metrik-metrik evaluasi ini memungkinkan kita memantau relevansi, akurasi, dan konsistensi hasil rekomendasi yang diberikan.

### Solution Approach

1. **Content-based Filtering**  
   Sistem ini menggunakan informasi deskripsi dan kategori untuk membuat rekomendasi berdasarkan karakteristik destinasi. Berikut ini adalah tahapan untuk mencapai tujuan ini:
   - **Preprocessing**  
     Proses pembersihan data teks dengan menggabungkan deskripsi dan kategori, lalu mengubahnya ke dalam bentuk numerik menggunakan metode TF-IDF Vectorizer.
   - **Pembentukan Model**  
     Setelah preprocessing, sistem menghitung kesamaan antar destinasi wisata menggunakan cosine similarity dari vektor TF-IDF untuk mengidentifikasi kesamaan antar destinasi.
   - **Evaluasi**  
     Metrik evaluasi yang digunakan untuk sistem content-based filtering adalah Precision@10, yang mengevaluasi proporsi dari 10 rekomendasi teratas yang relevan untuk pengguna.

2. **Collaborative Filtering**  
   Sistem ini memanfaatkan data rating yang diberikan pengguna untuk memberikan rekomendasi berdasarkan pola kesukaan pengguna lain yang serupa. Berikut ini adalah tahapan untuk mencapai tujuan ini:
   - **Encoding Data**  
     User ID dan Place ID diubah menjadi indeks numerik agar bisa digunakan pada model neural network.
   - **Normalisasi Rating**  
     Rating dinormalisasi agar berada dalam rentang antara 0 dan 1, sehingga model bisa belajar lebih optimal.
   - **Modeling**  
     Model neural network dikembangkan dengan embedding layer untuk pengguna dan destinasi, dengan hasil akhir berupa prediksi rating untuk setiap kombinasi pengguna dan destinasi.
   - **Evaluasi**  
     Metrik evaluasi yang digunakan untuk collaborative filtering meliputi RMSE dan MAE, yang mengukur tingkat perbedaan antara rating prediksi dan rating aktual.

## Data Understanding

### Informasi Dataset

Link Dataset: https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

Proyek ini menggunakan dataset Indonesia Tourism Destination yang dapat diakses melalui [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination). Dataset ini merupakan dataset yang berisi beberapa tempat wisata di 5 kota besar di Indonesia yaitu Jakarta, Yogyakarta, Semarang, Bandung, Surabaya. Data yang digunakan dalam proyek ini berasal dari dua file, yakni `tourism_with_id.csv` dan `tourism_rating.csv`.  
File `tourism_with_id.csv` berisi informasi tempat wisata di 5 kota di Indonesia, dengan total 437 data.  
File `tourism_rating.csv` berisi 10.000 data pengguna yang memberi rating ke destinasi wisata tertentu.

### Deskripsi Atribut Dataset

Berikut ini adalah deskripsi lengkap dari setiap atribut dalam dataset:

**1. File `tourism_with_id.csv`**

| No. | Kode Atribut | Deskripsi                                      | Tipe Data |
|-----|--------------|------------------------------------------------|-----------|
| 1   | Place_Id     | Identifikasi unik untuk setiap destinasi wisata.| Integer   |
| 2   | Place_Name   | Nama destinasi wisata.                         | String    |
| 3   | Description  | Deskripsi singkat tentang destinasi.           | String    |
| 4   | Category     | Kategori destinasi (misalnya: Budaya, Alam).   | String    |
| 5   | City         | Kota tempat destinasi berada.                  | String    |
| 6   | Price        | Harga tiket masuk.                             | Integer   |
| 7   | Rating       | Rating rata-rata dari pengguna (0-5).                | Float     |
| 8   | Time_Minutes | Waktu yang dibutuhkan untuk mengunjungi (dalam menit).       | Float     |
| 9   | Coordinate   | Koordinat geografis.                           | String    |
| 10  | Lat          | Latitude lokasi.                               | Float     |
| 11  | Long         | Longitude lokasi.                              | Float     |
| 12  | Unnamed: 11  | Kolom kosong.                                  | Float     |
| 13  | Unnamed: 12  | Nilai duplikat dari Place_id                                  | Integer   |
  
**2. File `tourism_rating.csv`**  

| No. | Kode Atribut  | Deskripsi                                                   | Tipe Data |
|-----|---------------|-------------------------------------------------------------|-----------|
| 1   | User_Id       | Identifikasi unik untuk setiap pengguna                    | Integer   |
| 2   | Place_Id      | Identifikasi unik untuk setiap destinasi wisata             | Integer   |
| 3   | Place_Ratings | Rating yang diberikan oleh pengguna kepada destinasi wisata  | Integer (0-5)   |

**Keterangan tipe data:**
- **Integer**: Angka bulat tanpa desimal.
- **Float**: Angka dengan desimal.
- **String**: Teks atau karakter.

**Catatan:**
- **Place_Id** pada dataset kedua ini berfungsi sebagai **foreign key** yang menghubungkan ke **Place_Id** pada dataset pertama (`tourism_with_id.csv`). Hal ini memungkinkan integrasi data antara informasi destinasi wisata dan rating pengguna.
- **User_Id** memungkinkan analisis perilaku pengguna secara individual serta pengelompokan pengguna berdasarkan preferensi mereka.

### Exploratory Data Analysis - EDA

Exploratory Data Analysis (EDA) adalah tahap awal dalam analisis data yang bertujuan untuk memahami struktur, pola, dan hubungan antar variabel dalam dataset. EDA membantu dalam mengidentifikasi anomali, distribusi data, serta insight awal yang dapat digunakan untuk proses persiapan data dan pemodelan selanjutnya. Dalam proyek ini, EDA dilakukan untuk memahami karakteristik destinasi wisata, distribusi rating, dan interaksi antara pengguna dengan destinasi wisata tertentu. Langkah-langkah EDA meliputi pemeriksaan struktur data, visualisasi distribusi variabel, dan analisis hubungan antar fitur yang relevan.

**Informasi Dataset `tourism_with_id.csv`**

**Struktur Data dengan `data_tourism_with_id.info()`**  

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 437 entries, 0 to 436
Data columns (total 13 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Place_Id      437 non-null    int64  
 1   Place_Name    437 non-null    object 
 2   Description   437 non-null    object 
 3   Category      437 non-null    object 
 4   City          437 non-null    object 
 5   Price         437 non-null    int64  
 6   Rating        437 non-null    float64
 7   Time_Minutes  205 non-null    float64
 8   Coordinate    437 non-null    object 
 9   Lat           437 non-null    float64
 10  Long          437 non-null    float64
 11  Unnamed: 11   0 non-null      float64
 12  Unnamed: 12   437 non-null    int64  
dtypes: float64(5), int64(3), object(5)
memory usage: 44.5+ KB
```

**Dari tampilan di atas terlihat bahwa:**

- Dataset ini terdiri dari **437 entri** (baris) dan **13 kolom**.

- Terdapat tiga tipe data utama dalam dataset:
  - **Integer** (`int64`): `Place_Id`, `Price`, dan `Unnamed: 12`.
  - **Float** (`float64`): `Rating`, `Time_Minutes`, `Lat`, `Long`, dan `Unnamed: 11`.
  - **String** (`object`): `Place_Name`, `Description`, `Category`, `City`, dan `Coordinate`.

- Nilai null:
  - Kolom `Time_Minutes` memiliki **205 nilai non-null** dari total 437 entri, yang berarti terdapat **232 nilai yang hilang**.
  - Kolom `Unnamed: 11` tidak memiliki nilai yang terisi (**0 non-null**), sehingga kolom ini sepenuhnya kosong.
  - Kolom lainnya memiliki **437 nilai non-null**, artinya tidak ada nilai yang hilang kecuali pada `Time_Minutes` dan `Unnamed: 11`.
- Dataset ini menggunakan **44.5 KB** memori.

**Contoh Data dengan `data_tourism_with_id.head()`**

![Informasi Dataset](/img/3.1.png)

**Informasi Dataset `tourism_rating.csv`**

**Struktur Data dengan `data_tourism_rating.info()`**

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   User_Id        10000 non-null  int64
 1   Place_Id       10000 non-null  int64
 2   Place_Ratings  10000 non-null  int64
dtypes: int64(3)
memory usage: 234.5 KB
```

**Dari tampilan di atas terlihat bahwa:**

- Dataset ini terdiri dari **10.000 entri** (baris) dan **3 kolom**.

- Semua kolom memiliki tipe data **Integer** (`int64`):
  - `User_Id`: Identifikasi unik untuk setiap pengguna.
  - `Place_Id`: Identifikasi unik untuk setiap destinasi wisata.
  - `Place_Ratings`: Rating yang diberikan oleh pengguna kepada destinasi wisata.

- Tidak ada nilai yang hilang dalam dataset ini, semua kolom memiliki **10.000 nilai non-null**.

- Dataset ini menggunakan **234.5 KB** memori.

**Contoh Data dengan `data_tourism_rating.head()`**

![Informasi Dataset](/img/3.2.png)

**Distribusi Rating Destinasi Wisata**

Dengan menggunakan `sns.histplot()`, didapatkan visualisasi distribusi rating destinasi wisata sebagai berikut.

![Distribusi Rating Destinasi Wisata](/img/3.3.png)

**Insight:**  
Distribusi rating destinasi wisata ini menunjukkan bahwa sebagian besar destinasi memiliki rating sekitar 4.4, menandakan kepuasan yang tinggi dari pengunjung. Distribusi agak condong ke kanan, dengan sedikit destinasi yang memiliki rating sangat tinggi (mendekati 5.0) atau rendah (di bawah 4.0). Hal ini mencerminkan kualitas yang konsisten dan pengalaman positif bagi wisatawan di sebagian besar destinasi.

**Jumlah Destinasi Wisata per Kategori**

Dengan menggunakan `sns.countlot()`, didapatkan grafik jumlah destinasi wisataper kategori sebagai berikut.

![Jumlah Destinasi Wisata per Kategori](/img/3.4.png)

**Insight:**  
Grafik ini menunjukkan jumlah destinasi wisata per kategori. Taman Hiburan memiliki jumlah terbanyak, diikuti oleh Budaya dan Cagar Alam. Kategori Bahari cukup signifikan, namun Tempat Ibadah dan Pusat Perbelanjaan memiliki jumlah yang paling sedikit. Hal ini membuat opsi wisata cenderung lebih kepada taman hiburan, budaya, dan alam, dibandingkan kategori lainnya.

**Jumlah Rating per Destinasi Wisata (Top 10)**  

Dengan menggunakan `sns.barlot()`, didapatkan grafik jumlah rating per destinasi wisata sebagai berikut.

![Jumlah Rating per Destinasi Wisata (Top 10)](/img/3.5.png)

**Insight:**  
Jumlah rating yang tinggi pada destinasi tertentu, seperti yang terlihat pada Place_ID 437, 177, dan 298, menunjukkan bahwa destinasi ini tidak hanya sering dikunjungi tetapi juga mungkin memiliki daya tarik yang kuat di mata wisatawan. Jumlah rating yang tinggi umumnya mencerminkan tingkat engagement yang lebih besar, di mana wisatawan merasa terdorong untuk memberikan penilaian mereka, baik positif maupun negatif. Destinasi yang populer ini bisa memiliki fasilitas yang baik, pengalaman unik, atau tingkat promosi yang lebih tinggi dibandingkan destinasi lain. Selain itu, tingginya jumlah rating juga bisa mengindikasikan reputasi destinasi tersebut.

## Data Preparation

Pada tahap **data preparation**, dilakukan serangkaian langkah untuk memastikan bahwa data yang digunakan dalam proses pemodelan bersih, terstruktur, dan siap untuk dianalisis. Tahapan ini mencakup pembersihan data, penggabungan dataset, pengolahan teks, encoding variabel kategorikal, normalisasi rating, dan pembagian data menjadi set pelatihan dan pengujian. Berikut adalah uraian mendetail mengenai setiap langkah yang dilakukan:

### 1. Mengatasi Missing Values dan Data yang Tidak Diperlukan

**a. Identifikasi Missing Values**

Dari analisis sebelumnya menggunakan `data_tourism_with_id.info()`, diketahui bahwa:

- Kolom `Time_Minutes` memiliki **205 nilai non-null** dari total 437 entri, yang berarti terdapat **232 nilai yang hilang**.
- Kolom `Unnamed: 11` tidak memiliki nilai yang terisi (**0 non-null**), sehingga kolom ini sepenuhnya kosong.

**b. Penanganan Missing Values**

Kedua kolom `Time_Minutes` dan `Unnamed: 11` dihapus karena kedua kolom memang tidak diperlukan untuk model Content-based Filtering dan Collaborative Filtering dalam konteks rekomendasi destinasi wisata.

**c. Menghapus Kolom yang Tidak Perlu**

- Kolom `Unnamed: 11` dan `Unnamed: 12`dihapus karena `Unnamed: 11` sepenuhnya kosong dan `Unnamed: 12` juga tidak relevan.
- Kolom `City`, `Price`, `Time_Minutes`, `Coordinate`, `Lat`, dan `Long` dihapus karena sistem rekomendasi akan dibangun berdasarkan kategori dan deskripsi, bukan informasi dari kolom-kolom tersebut.

```python
columns_to_drop = ['City', 'Price', 'Time_Minutes', 'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12']
data_tourism_with_id_clean = data_tourism_with_id.drop(columns=columns_to_drop, axis=1, errors='ignore')
```

### 2. Menggabungkan Dataset

- **Cara**:  
  Dataset `tourism_rating.csv` digabungkan dengan `tourism_with_id_clean` berdasarkan `Place_Id` untuk memperoleh informasi lengkap tentang setiap destinasi beserta rating rata-rata.

  ```python
  recommendation_data = pd.merge(
      data_tourism_rating.groupby('Place_Id')['Place_Ratings'].mean().reset_index(),
      data_tourism_with_id_clean,
      on='Place_Id'
  )
  ```
  
- **Alasan**:  
  Menggabungkan kedua dataset memungkinkan model untuk memanfaatkan informasi detail tentang destinasi wisata serta interaksi pengguna melalui rating yang diberikan.

### 3. Mengisi Nilai Hilang pada Atribut Teks

Untuk memastikan konsistensi data teks, nilai yang hilang pada kolom `Description` dan `Category` diisi dengan string kosong.

```python
recommendation_data['Description'] = recommendation_data['Description'].fillna('')
recommendation_data['Category'] = recommendation_data['Category'].fillna('')
```

**Alasan:**  
Mengisi nilai yang hilang dengan string kosong memungkinkan proses preprocessing teks berjalan lancar tanpa mengganggu analisis selanjutnya.

### 4. Pemeriksaan Duplikasi Data

Untuk menjaga integritas data, dilakukan pemeriksaan terhadap duplikasi pada atribut `Place_Name`.

```python
if recommendation_data['Place_Name'].duplicated().any():
```

**Hasil:**  
Tidak ditemukan data duplikasi

### 5. Preprocessing Data untuk Content-based Filtering

**a. Penggabungan Deskripsi dan Kategori Menjadi Satu Teks**  

Untuk memanfaatkan informasi deskripsi dan kategori dalam sistem rekomendasi berbasis konten, kedua atribut ini digabungkan menjadi satu atribut baru bernama `Tags`.

```python
recommendation_data['Tags'] = recommendation_data['Description'] + ' ' + recommendation_data['Category']
```

**b. Pembersihan Teks (Preprocessing)**  

Teks pada atribut `Tags` dan `Description` dibersihkan dengan langkah-langkah berikut:

1. **Lowercasing:** Mengubah semua huruf menjadi huruf kecil untuk konsistensi.
2. **Stemming:** Mengurangi kata ke bentuk dasarnya menggunakan library Sastrawi.
3. **Stopword Removal:** Menghapus kata-kata yang tidak memiliki makna signifikan (stopwords).

```python
recommendation_data['Tags'] = recommendation_data['Tags'].apply(preprocessing)
recommendation_data['Description_Preprocessed'] = recommendation_data['Description'].apply(preprocessing)
```

**Isi Fungsi Preprocessing:**

```python
def preprocessing(text):
    text = text.lower()
    text = stemmer.stem(text)
    text = stopword.remove(text)
    return text
```

**Alasan:**
- **Lowercasing** dilakukan untuk menghindari duplikasi kata yang sama namun berbeda kapitalisasi.
- **Stemming dan Stopword Removal** dilakukan untuk mengurangi kompleksitas teks dan meningkatkan relevansi fitur teks yang digunakan dalam vektorisasi.

**c. Vektorisasi Teks dengan TF-IDF Vectorizer**  

Setelah pembersihan teks, teks diubah menjadi representasi numerik menggunakan `TfidfVectorizer`. Dua atribut vektorisasi dilakukan:

1. **Tags (Deskripsi + Kategori):**

   ```python
   vectors_tags = tv_tags.fit_transform(recommendation_data['Tags'])
   ```

2. **Description Saja:**

   ```python
   vectors_desc = tv_desc.fit_transform(recommendation_data['Description_Preprocessed'])
   ```

**Alasan:**  
**TF-IDF Vectorizer** dapat mengubah teks menjadi vektor numerik yang memungkinkan pengukuran kesamaan antar destinasi menggunakan metrik cosine similarity.

**d. Menghitung Cosine Similarity**

Untuk mengukur kesamaan antar destinasi wisata, dihitung cosine similarity berdasarkan vektor TF-IDF yang telah dibuat.

1. **Similarity Berdasarkan Tags:**

   ```python
   similarity_tags = cosine_similarity(vectors_tags, dense_output=False)
   ```

2. **Similarity Berdasarkan Description:**

   ```python
   similarity_desc = cosine_similarity(vectors_desc, dense_output=False)
   ```

**Alasan:**  
Metrik ini efektif untuk mengukur kesamaan antar dokumen teks, sehingga cocok untuk sistem rekomendasi berbasis konten.

### 6. Persiapan Data untuk Collaborative Filtering

**a. Encoding User_Id dan Place_Id**

Untuk keperluan pemodelan, `User_Id` dan `Place_Id` diubah menjadi indeks numerik.

```python
user_ids = data_tourism_rating['User_Id'].unique().tolist()
place_ids = data_tourism_rating['Place_Id'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

data_collab = data_tourism_rating.copy()
data_collab['user'] = data_collab['User_Id'].map(user_to_user_encoded)
data_collab['place'] = data_collab['Place_Id'].map(place_to_place_encoded)
```

**Alasan:**  
Mengubah ID ke indeks numerik memungkinkan penggunaan embedding layers dalam model neural network untuk Collaborative Filtering.

**b. Normalisasi Rating**

Rating yang diberikan oleh pengguna dinormalisasi agar berada dalam rentang antara 0 dan 1.

```python
min_rating = data_collab['Place_Ratings'].min()
max_rating = data_collab['Place_Ratings'].max()
data_collab['normalized_rating'] = data_collab['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
```

**Alasan:**
**Normalisasi** membantu model neural network dalam proses pelatihan dengan memastikan bahwa nilai input berada dalam rentang yang seragam, sehingga mempercepat konvergensi dan meningkatkan performa model.

**c. Membagi Data Menjadi Train dan Test**

Data dibagi menjadi data latih (train) dan data validasi (validation) untuk evaluasi model.

```python
train_size_cf = int(0.8 * len(data_collab))
x_train_cf = data_collab[['user', 'place']].values[:train_size_cf]
y_train_cf = data_collab['normalized_rating'].values[:train_size_cf]
x_val_cf = data_collab[['user', 'place']].values[train_size_cf:]
y_val_cf = data_collab['normalized_rating'].values[train_size_cf:]
```

**Alasan:**  
Memisahkan data menjadi train dan test memungkinkan evaluasi performa model pada data yang belum pernah dilihat selama pelatihan, sehingga memberikan indikasi yang lebih akurat mengenai generalisasi model.

## Modeling and Result

Pada tahap ini, dua sistem rekomendasi yang berbeda, yaitu **Content-based Filtering** dan **Collaborative Filtering** dibangun dan dievaluasi. Masing-masing pendekatan memiliki metode dan algoritma yang unik untuk memberikan rekomendasi destinasi wisata kepada pengguna.

### 1. Content-based Filtering

**Content-based Filtering** menggunakan informasi konten dari destinasi wisata, seperti deskripsi dan kategori, untuk memberikan rekomendasi yang relevan kepada pengguna. Pendekatan ini menganalisis kesamaan antara destinasi wisata berdasarkan atribut-atribut tersebut.

**Langkah-langkah:**

- Memastikan nama destinasi wisata (`place_name`) ada dalam data.
- Mendapatkan indeks destinasi wisata yang diminta. Indeks digunakan untuk mengakses matriks similarity.
- Menghitung skor kesamaan (similarity score) untuk destinasi tersebut. Proses ini menggunakan matriks similarity yang telah dihitung sebelumnya.
- Mengurutkan destinasi berdasarkan skor kesamaan.
- Menghindari merekomendasikan destinasi yang sama dengan yang diminta. Hal ini dilakukan dengan menghapus indeks destinasi tersebut dari daftar destinasi yang direkomendasikan.
- Mendapatkan top-n rekomendasi (misalnya top 10).

**Implementasi Kode:**

```python
def get_content_based_recommendations(place_name, data, similarity_matrix, top_n=10):
    """
    Mengembalikan rekomendasi Content-based Filtering untuk sebuah destinasi wisata.

    Parameters:
    - place_name (str): Nama destinasi wisata.
    - data (DataFrame): Dataset destinasi wisata (recommendation_data).
    - similarity_matrix (ndarray): Matriks similarity antar destinasi.
    - top_n (int): Jumlah rekomendasi yang diinginkan.

    Returns:
    - list: Daftar Place_Id yang direkomendasikan.
    """
    # Memastikan place_name ada dalam data
    if place_name not in data['Place_Name'].values:
        print(f"'{place_name}' TIDAK ditemukan dalam dataset.")
        return []

    # Mendapatkan indeks destinasi yang diminta
    place_idx = data[data['Place_Name'] == place_name].index[0]

    # Mendapatkan skor similarity untuk destinasi tersebut dan mengonversi ke dense array
    place_similarity = similarity_matrix[place_idx].toarray().flatten()

    # Mengurutkan destinasi berdasarkan similarity score
    similar_indices = place_similarity.argsort()[::-1]

    # Menghindari rekomendasi diri sendiri
    similar_indices = similar_indices[similar_indices != place_idx]

    # Mendapatkan top-n rekomendasi
    top_indices = similar_indices[:top_n]
    recommended_place_ids = data.iloc[top_indices]['Place_Id'].tolist()

    return recommended_place_ids
```

**Hasil Rekomendasi:**

Sebagai contoh, kami melakukan rekomendasi untuk destinasi wisata **'Wisata Alam Kalibiru'**.

| No. | Place Name                         | Category      | Rating | Description                                                                             |
|-----|------------------------------------|---------------|--------|-----------------------------------------------------------------------------------------|
| 1   | Watu Lumbung                       | Cagar Alam    | 4.3    | Letak Kampung Edukasi Watu Lumbung yang berada...                                        |
| 2   | Wisata Kaliurang                   | Cagar Alam    | 4.4    | Jogja selalu menarik untuk dikulik, terlebih t...                                        |
| 3   | Ciwangun Indah Camp Official       | Cagar Alam    | 4.3    | Ciwangun Indah Camp atau CIC adalah sebuah tem...                                        |
| 4   | Curug Cilengkrang                  | Cagar Alam    | 4.0    | Curug Cilengkrang bisa menjadi pilihan tujuan ...                                        |
| 5   | Happyfarm Ciwidey                  | Cagar Alam    | 4.2    | Objek wisata alam dan edukasi tengah banyak me...                                        |
| 6   | Hutan Wisata Tinjomoyo Semarang    | Cagar Alam    | 4.3    | Awalnya taman wisata hutan Tinjomoyo Semarang ...                                        |
| 7   | Umbul Sidomukti                    | Cagar Alam    | 4.6    | Kawasan wisata umbul Sidomukti merupakan salah...                                        |
| 8   | Wisata Alam Wana Wisata Penggaron  | Cagar Alam    | 4.1    | Berada sekitar 2 KM dari Kota Ungaran atau sek...                                        |
| 9   | Kampoeng Kopi Banaran              | Taman Hiburan | 4.3    | Kampoeng Kopi Banaran, sebuah agro wisata perk...                                       |
| 10  | Air Terjun Semirang                | Cagar Alam    | 4.4    | Terletak di lereng Gunung Ungaran bagian utara...                                       |

**Kelebihan dan Kekurangan:**

| **Kelebihan**                                                                 | **Kekurangan**                                                            |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Dapat memberikan rekomendasi yang spesifik berdasarkan preferensi konten.  | Memerlukan data deskripsi dan kategori yang cukup banyak dan berkualitas.      |
| Tidak memerlukan data interaksi pengguna (*rating*).                         | Rekomendasi terbatas pada konten yang sudah ada dalam dataset.         |
| Cocok untuk pengguna baru (*cold start*) yang belum memiliki interaksi.       | Rentan terhadap overfitting jika data konten tidak bervariasi.          |

### 2. Collaborative Filtering

**Collaborative Filtering** memanfaatkan data interaksi pengguna, seperti rating yang diberikan, untuk memberikan rekomendasi berdasarkan pola preferensi pengguna lain yang serupa. Pendekatan ini dapat menangkap preferensi yang lebih kompleks dan dinamis.

**Langkah-langkah:**

1. **Membangun Model Neural Network**  
   - Menggunakan embedding layers untuk pengguna dan destinasi wisata.
   - Mengombinasikan embedding dengan bias dan menghitung dot product.
   - Menggunakan *sigmoid activation* untuk output prediksi rating.

5. **Melatih Model**  
   Melatih model selama 20 epoch dengan batch size 32 menggunakan optimizer Adam.

6. **Fungsi Rekomendasi**  
   Membuat fungsi `recommend_by_collaborative_filtering` yang mengambil `User_Id` dan mengembalikan daftar rekomendasi berdasarkan prediksi rating tertinggi.

**Implementasi Kode:**

```python
class TourismRecNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        """
        Inisialisasi model Collaborative Filtering.

        Parameters:
        - num_users (int): Jumlah pengguna.
        - num_places (int): Jumlah tempat wisata.
        - embedding_size (int): Ukuran embedding untuk pengguna dan tempat.
        """
        super(TourismRecNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.place_embedding = layers.Embedding(num_places, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.place_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        """
        Melakukan forward pass melalui model.

        Parameters:
        - inputs (tf.Tensor): Input tensor dengan shape (batch_size, 2)

        Returns:
        - tf.nn.sigmoid: Prediksi rating yang dihasilkan.
        """
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])

        dot_product = tf.tensordot(user_vector, place_vector, 2)
        x = dot_product + user_bias + place_bias
        return tf.nn.sigmoid(x)

# Inisialisasi dan compile model dengan RMSE sebagai loss function
model = TourismRecNet(num_users, num_places, 50)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Melatih model
history = model.fit(
    x=x_train_cf,
    y=y_train_cf,
    batch_size=32,
    epochs=20,
    validation_data=(x_val_cf, y_val_cf)
)
```

**Hasil Pelatihan Model:**

**Plot RMSE selama Pelatihan:**

![Model RMSE](/img/5.1.png)  

Grafik menunjukkan penurunan nilai RMSE pada data train dan validation seiring bertambahnya epoch, namun penurunannya tidaklah signifikan. Maka, epoch tambahan dirasa tidak diperlukan lagi.

**Hasil Rekomendasi:**

Sebagai contoh, kami melakukan rekomendasi untuk seorang pengguna dengan `User_Id` terpilih.

Rekomendasi berdasarkan Collaborative Filtering untuk User ID 27:

| No. | Place Name                    | Category      | Rating | Description                                                                             |
|-----|-------------------------------|---------------|--------|-----------------------------------------------------------------------------------------|
| 1   | Pantai Baron                  | Bahari        | 4.4    | Pantai Baron adalah salah satu objek wisata be...                                        |
| 2   | Pintoe Langit Dahromo         | Cagar Alam    | 4.4    | Pintu Langit Dahromo ini menyediakan berbagai ...                                        |
| 3   | La Kana Chapel                | Taman Hiburan | 4.5    | La Kana Chapel menawarkan konsep baru standing...                                        |
| 4   | Goa Rong                      | Cagar Alam    | 4.3    | Semarang memiliki wisata di ketinggian bernama...                                        |
| 5   | Kampoeng Kopi Banaran         | Taman Hiburan | 4.3    | Kampoeng Kopi Banaran, sebuah agro wisata perk...                                       |
| 6   | Monumen Kapal Selam           | Budaya        | 4.4    | Monumen Kapal Selam, atau disingkat Monkasel, ...                                        |
| 7   | Taman Keputran                | Taman Hiburan | 4.3    | Ntah, mengapa nama taman ini disebut dengan ta...                                        |
| 8   | Taman Ekspresi Dan Perpustakaan | Taman Hiburan | 4.5    | Taman Ekspresi Surabaya tidak hanya menyuguhka...                                        |
| 9   | Keraton Surabaya              | Budaya        | 4.4    | Kawasan yang berjuluk Kampung Keraton ini terd...                                        |
| 10  | Taman Hiburan Rakyat          | Taman Hiburan | 4.2    | Taman Hiburan Rakyat atau THR tentunya sudah t...                                        |

**Kelebihan dan Kekurangan**

| **Kelebihan**                                                                 | **Kekurangan**                                                            |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Dapat menangkap data yang lebih kompleks dan dinamis.                 | Memerlukan data interaksi pengguna yang cukup untuk model belajar.     |
| Mampu memberikan rekomendasi yang lebih personal dan relevan.               | Rentan terhadap masalah *sparsity* jika data interaksi pengguna rendah.   |
| Dapat mengidentifikasi pola tersembunyi dalam data interaksi pengguna.      | Tidak cocok untuk pengguna baru yang belum memiliki interaksi (*cold start*). |

## Evaluasi

Evaluasi merupakan langkah untuk menilai sejauh mana performa sistem rekomendasi yang telah dibangun memenuhi tujuan yang telah ditetapkan. Dalam proyek ini, terdapat tiga metrik evaluasi utama yaitu:
- **Precision** untuk menilai pendekatan sistem **Content-based Filtering**
- **Root Mean Squared Error (RMSE)** serta **Mean Absolute Error (MAE)** untuk menilai pendekatan sistem rekomendasi **Collaborative Filtering**.

### 1. Metrik Evaluasi yang Digunakan

**a. Precision**  
Precision mengukur proporsi rekomendasi yang relevan di antara 10 rekomendasi teratas yang diberikan kepada pengguna. Metrik ini memberikan indikasi seberapa akurat rekomendasi yang dihasilkan oleh sistem.

**Formula:**  
$$\text{Precision} = \frac{\text{TP}}{TP+FP}$$  

Di mana:
- $TP$ (True Positive) adalah jumlah rekomendasi yang relevan di antara 10 rekomendasi teratas,
- $FP$ (False Positive) adalah jumlah rekomendasi yang tidak relevan di antara 10 rekomendasi teratas.

**Interpretasi:**
- Jika nilai precision tinggi, artinya sebagian besar rekomendasi yang diberikan relevan dengan preferensi pengguna.
- Jika nilai precision rendah, artinya sebagian besar rekomendasi yang diberikan tidak relevan.

Dalam implementasi kode, alur untuk menentukan jumlah rekomendasi yang relevan (TP) dan kebalikannya (FP) adalah sebagai berikut:

**Langkah-langkah Algoritma:**

- Algoritma memilih satu atau beberapa destinasi wisata (`places_to_evaluate`) yang akan dievaluasi kualitas rekomendasinya.

- Untuk setiap destinasi wisata yang dipilih (`place`), algoritma menggunakan fungsi `get_content_based_recommendations` untuk menghasilkan daftar 10 rekomendasi destinasi wisata teratas (`recommended_place_ids`) berdasarkan kesamaan konten (deskripsi dan kategori).

- Dari daftar rekomendasi yang dihasilkan, algoritma mengambil informasi detail mengenai setiap destinasi wisata yang direkomendasikan, seperti `Place_Name`, `Category`, `Rating`, dan `Description_Preprocessed`.

- Algoritma menghitung kesamaan deskripsi (`Description_Similarity`) antara destinasi wisata input (`place`) dan setiap destinasi yang direkomendasikan menggunakan matriks kesamaan deskripsi (`similarity_desc`).

- **Menentukan Relevansi Rekomendasi**  
  Setiap rekomendasi dievaluasi apakah relevan atau tidak berdasarkan dua kriteria:
     - **Kecocokan Kategori:** Jika kategori destinasi wisata yang direkomendasikan sama dengan kategori destinasi input.
     - **Kesamaan Deskripsi:** Jika kesamaan deskripsi antara destinasi input dan destinasi yang direkomendasikan melebihi ambang batas tertentu (`desc_threshold`), misalnya 0.5.  

  Jika salah satu dari kedua kriteria tersebut terpenuhi, rekomendasi dianggap relevan (`Relevance = 1`), sebaliknya tidak relevan (`Relevance = 0`).

- **Menghitung True Positives (TP) dan False Positives (FP)**
   - **True Positives (TP):** Jumlah rekomendasi yang relevan (Relevance = 1) dari total 10 rekomendasi.
   - **False Positives (FP):** Jumlah rekomendasi yang tidak relevan, dihitung sebagai `top_n - TP`.

- **Menghitung Precision** dengan rumus yang telah disebutkan sebelumnya.

**b. Root Mean Squared Error (RMSE)**

RMSE mengukur rata-rata perbedaan antara nilai prediksi model dan nilai aktual yang diamati, dengan memberikan bobot lebih pada kesalahan yang lebih besar.

**Formula:**  
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$  
Di mana:
- $y_i$ adalah nilai aktual,
- $\hat{y}_i$ adalah nilai prediksi,
- $n$ adalah jumlah sampel.
- $i$ adalah indeks sampel.

**Interpretasi:**
- Jika nilai RMSE rendah, berarti prediksi model mendekati nilai aktual.
- Jika nilai RMSE tinggi, berarti ada kesalahan prediksi yang signifikan.

**c. Mean Absolute Error (MAE)**

MAE mengukur rata-rata kesalahan absolut antara nilai prediksi model dan nilai aktual yang diamati.

**Formula:**  
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$  

Di mana:
- $y_i$ adalah nilai aktual,
- $\hat{y}_i$ adalah nilai prediksi,
- $n$ adalah jumlah sampel.

**Interpretasi:**
- Jika nilai MAE rendah, artinya model memiliki akurasi prediksi yang baik.
- Jika nilai MAE tinggi, artinya terdapat kesalahan prediksi yang besar secara konsisten.

### 2. Hasil Evaluasi

**a. Evaluasi Precision untuk Content-based Filtering**

Evaluasi Precision berhasil dilakukan untuk beberapa destinasi wisata yang dipilih, misalnya **'Museum Fatahillah'**. Hasilnya adalah:

| Place               | Precision@10 |
|---------------------|--------------|
| Museum Fatahillah   | 100.00%      |

Precision sebesar **100.00%** menunjukkan bahwa 10 dari 10 rekomendasi yang diberikan kepada pengguna terkait **'Museum Fatahillah'** relevan. Artinya, performanya baik dalam memberikan rekomendasi yang sesuai berdasarkan konten deskripsi dan kategori.

**b. Evaluasi RMSE dan MAE untuk Collaborative Filtering**

Performa model Collaborative Filtering dievaluasi pada data validasi dengan menghitung nilai **RMSE** dan **MAE**. Hasilnya adalah:

```
Collaborative Filtering - RMSE: 0.3583
Collaborative Filtering - MAE: 0.3105
```

**Interpretasi:**
- Nilai RMSE sebesar **0.3583** menunjukkan bahwa rata-rata model memiliki performa yang cukup baik dalam memprediksi rating pengguna. Namun, masih terdapat ruang untuk meningkatkan akurasi prediksi melalui tuning hyperparameter dan/atau eksperimen dengan arsitektur model yang berbeda.
- Nilai MAE sebesar **0.3105** juga menunjukkan tingkat akurasi yang baik dalam prediksi rating, meskipun terdapat ruang untuk peningkatan.

### 3. Visualisasi Hasil Evaluasi

**Evaluasi RMSE dan MAE untuk Collaborative Filtering**

![Metrik Evaluasi Collaborative Filtering](/img/5.2.png)

**Interpretasi:**  
Grafik batang menunjukkan nilai RMSE dan MAE yang cukup rendah, menandakan bahwa model Collaborative Filtering mampu memprediksi rating dengan baik.

### 4. Evaluasi Keseluruhan

Hasil evaluasi terhadap kedua model rekomendasi, yaitu **Content-based Filtering** dan **Collaborative Filtering** menunjukkan bahwa model yang dibuat sudah mencapai tujuan yang diharapkan.

Pertama-tama, model **Content-based Filtering** berhasil memberikan rekomendasi yang relevan pada data **Museum Fatahillah** dengan **Precision** sebesar **100.00%**. Hal ini menunjukkan bahwa mayoritas rekomendasi yang diberikan oleh model ini sesuai dengan preferensi pengguna berdasarkan konten deskripsi dan kategori destinasi wisata. Dengan demikian, permasalahan pertama yang menyangkut bagaimana memberikan rekomendasi destinasi wisata yang relevan berdasarkan konten deskripsi dan kategori telah terjawab dengan baik. Model ini mampu memanfaatkan informasi tekstual untuk mengenali kesamaan antar destinasi, sehingga memberikan rekomendasi yang tepat sasaran.

Di sisi lain, model **Collaborative Filtering** menunjukkan performa yang cukup baik dengan nilai **RMSE** sebesar **0.3583** dan **MAE** sebesar **0.3105**. Meskipun demikian, masih terdapat ruang untuk peningkatan agar model dapat lebih mendekati prediksi yang sempurna. Terlepas dari itu, model ini berhasil menangkap pola preferensi pengguna berdasarkan data rating yang diberikan, sehingga menjawab permasalahan kedua mengenai bagaimana memanfaatkan data rating pengguna untuk meningkatkan akurasi rekomendasi. Dengan mengidentifikasi pola interaksi pengguna, model ini dapat memberikan rekomendasi yang lebih personal dan relevan, meskipun masih membutuhkan optimasi lebih lanjut untuk meningkatkan akurasinya.

Selain itu, evaluasi menggunakan metrik **Precision**, **RMSE**, dan **MAE** telah memastikan bahwa sistem rekomendasi tidak hanya relevan tetapi juga akurat dalam memenuhi kebutuhan pengguna. Precision memberikan indikasi bahwa rekomendasi yang diberikan oleh model Content-based Filtering cukup relevan, sementara RMSE dan MAE pada model Collaborative Filtering menunjukkan bahwa prediksi rating pengguna cukup akurat. Dengan demikian, sistem rekomendasi ini telah berhasil mengukur relevansi dan akurasi rekomendasi, memenuhi tujuan ketiga yaitu mengukur performa sistem rekomendasi menggunakan metrik evaluasi yang sesuai.

Secara keseluruhan, kedua pendekatan sistem rekomendasi yang diimplementasikan—**Content-based Filtering** dan **Collaborative Filtering**—telah berhasil menjawab problem statements dan mencapai goals yang diharapkan. Model Content-based Filtering memberikan rekomendasi yang spesifik dan relevan berdasarkan konten, sementara Collaborative Filtering mampu memprediksi preferensi pengguna dengan baik melalui analisis pola interaksi. Dampak positif dari kedua model ini terhadap bisnis sangat terasa, karena mereka meningkatkan kemampuan sistem dalam memberikan rekomendasi yang akurat dan relevan, yang pada gilirannya dapat meningkatkan kepuasan dan loyalitas pengguna. Dengan demikian, solusi yang telah direncanakan dan diimplementasikan dalam proyek ini telah memberikan dampak yang positif terhadap pemahaman bisnis, memenuhi kebutuhan pengguna, serta mencapai tujuan yang diharapkan.

### 5. Saran

Terdapat beberapa area yang masih dapat ditingkatkan. Untuk model Collaborative Filtering, optimasi lebih lanjut seperti tuning hyperparameter atau eksplorasi arsitektur model yang berbeda dapat dilakukan untuk meningkatkan akurasinya. Selain itu, menggabungkan kedua pendekatan ini dalam sebuah **Hybrid Approach** dapat memberikan manfaat tambahan dengan menggabungkan kekuatan masing-masing metode, sehingga menghasilkan rekomendasi yang lebih akurat dan relevan secara keseluruhan.

## Referensi

1. Badan Pusat Statistik. (2020). *Jumlah Wisatawan Mancanegara ke Indonesia Tahun 2019*. Diambil dari [https://www.bps.go.id/id/pressrelease/2020/02/03/1711/jumlah-kunjungan-wisman-ke-indonesia-desember-2019-mencapai-1-38-juta-kunjungan-.html](https://www.bps.go.id/id/pressrelease/2020/02/03/1711/jumlah-kunjungan-wisman-ke-indonesia-desember-2019-mencapai-1-38-juta-kunjungan-.html)
2. World Tourism Organization. (2021). *Impact Assessment of the COVID-19 Outbreak on International Tourism*. Diambil dari [https://www.unwto.org/impact-assessment-of-the-covid-19-outbreak-on-international-tourism](https://www.unwto.org/impact-assessment-of-the-covid-19-outbreak-on-international-tourism)
3. Rezkia, S. M., & Wibowo, B. S. (2024). Pengembangan Sistem Rekomendasi Rencana Perjalanan: Literature Review. *Prosiding Seminar Nasional Teknik Industri (SeNTI)*.
4. Overbeek, M. V., & Naatonis, R. N. (2019). Sistem Rekomendasi Destinasi Wisata Di Kota Kupang Dengan Metode Weighted Product. *HOAQ (High Education of Organization Archive Quality): Jurnal Teknologi Informasi*, 10(1), 30-34.
5. Murzani, F. F., & Arianto, D. B. (2023). Implementasi Metode Collaborative Filtering pada Algoritma Sistem Rekomendasi Destinasi Wisata di Aceh. *Jurnal Komputer, Informasi Teknologi, dan Elektro*, 8(3).
6. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
7. Robertson, S. (2004). Understanding inverse document frequency: on theoretical arguments for IDF. *Journal of Documentation*, 60(5), 503-520. https://doi.org/10.1108/00220410410560582
8. Zhang, Y., Chen, C., & Wang, X. (2020). *Fast Adaptation for Cold-Start Collaborative Filtering with Meta-Learning*. Di dalam *2020 IEEE International Conference on Data Mining (ICDM)* (hal. 661-670). Sorrento, Italia: IEEE. https://doi.org/10.1109/ICDM50108.2020.00075
