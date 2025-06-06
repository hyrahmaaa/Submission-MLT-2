# Laporan Proyek Machine Learning - Rahayu Nur Rahmawati

## Project Overview

Dilihat    dari    banyaknya jumlah penonton bioskop yang terus  meningkat dari  tahun  ke  tahun.  Per  2018  angka  jumlah penonton   bioskop   di   Indonesia   saja   telah mencapai  lebih  dari50  juta  penonton  dengan jumlah  produksi  film  luar  negeri  hingga  dalam negeri  sebanyak  hampir  200  judul  film  yang telah  tayang  di  seluruh  Indonesia. Dengan semakin banyaknya pilihan film yang tersedia di berbagai platform streaming dan bioskop, pengguna seringkali kesulitan menemukan film yang sesuai dengan selera mereka. Fenomena ini, yang dikenal sebagai 'paradox of choice' (Schwartz, 2004) atau information overload, dapat menyebabkan pengguna menghabiskan waktu terlalu banyak untuk mencari tanpa hasil yang memuaskan, bahkan melewatkan film-film berkualitas tinggi yang sebenarnya relevan dengan minat mereka. Pergeseran perilaku menonton dari televisi tradisional ke layanan video-on-demand telah menciptakan katalog film yang masif, membuat navigasi dan penemuan konten baru menjadi tantangan tersendiri bagi penonton. 

Untuk mengatasi permasalahan tersebut, sistem rekomendasi telah menjadi solusi krusial dalam industri hiburan digital. Sistem ini bertujuan untuk mempersonalisasi pengalaman pengguna dengan menyarankan film berdasarkan preferensi, riwayat tontonan, atau kesamaan fitur antar film. Proyek ini hadir untuk mengembangkan sebuah model sistem rekomendasi film yang efektif, berfokus pada pendekatan Content-Based Filtering, untuk membantu pengguna menavigasi lautan konten dan menemukan film yang paling sesuai dengan preferensi mereka, sehingga meningkatkan kepuasan dan pengalaman menonton secara keseluruhan.

**Referensi :**: 
Schwartz, B. (2004). The Paradox of Choice: Why More Is Less. Ecco.
Zhang, J., Wen, J., & Sun, Q. (2019). Predicting Movie Box-Office Success: A Comparative Study of Machine Learning Algorithms. Neurocomputing, 321, 259–269. https://doi.org/10.1016/j.neucom.2018.09.027

## Business Understanding

### Problem Statements

Dengan semakin banyaknya pilihan film yang tersedia di berbagai platform streaming dan bioskop, pengguna seringkali kesulitan menemukan film yang sesuai dengan selera mereka. Hal ini dapat menyebabkan:
1. Waktu pencarian yang terbuang: Pengguna menghabiskan waktu terlalu banyak untuk mencari film tanpa hasil yang memuaskan.
2. Konten relevan terlewat: Pengguna melewatkan film-film berkualitas tinggi yang sebenarnya relevan dengan minat mereka.
   
### Goals

Untuk menjawab pertanyaan tersebut dan mengatasi permasalahan yang ada, proyek ini bertujuan untuk:
1. Mempercepat dan mempersonalisasi pencarian: Membangun model machine learning berbasis Content-Based Filtering yang mampu merekomendasikan film kepada pengguna berdasarkan kesamaan fitur film (genre dan aktor), sehingga mempercepat proses penemuan film yang relevan.
2. Meningkatkan penemuan konten relevan: Menganalisis dan memahami karakteristik serta preferensi film dari kumpulan data yang tersedia terutama berdasarkan user_rating untuk memastikan rekomendasi yang diberikan adalah film-film berkualitas tinggi dan sesuai dengan minat pengguna.

**Solution Approach**:

Untuk mencapai tujuan proyek dalam mengatasi kesulitan pengguna menemukan film yang relevan dan berkualitas tinggi, proyek ini mengadopsi pendekatan berbasis data dengan memanfaatkan kekuatan Machine Learning. Pendekatan solusi ini dirancang untuk secara sistematis mengidentifikasi preferensi konten film dan menyajikannya dalam format yang mudah diakses. Beberapa poin kunci dalam pendekatan solusi ini meliputi:
1. Metodologi Content-Based Filtering
Pemilihan Pendekatan: Sistem rekomendasi ini dibangun menggunakan metodologi Content-Based Filtering. Pendekatan ini dipilih karena efektivitasnya dalam merekomendasikan item berdasarkan atribut internalnya (seperti genre dan aktor), yang sangat cocok dengan jenis data film yang tersedia.
Basis Rekomendasi: Rekomendasi didasarkan pada kesamaan profil konten antar film. Jika pengguna menyukai Film A, sistem akan mencari dan merekomendasikan Film B yang memiliki karakteristik konten yang serupa.
2. Pengolahan Fitur Teks dan Vektorisasi Lanjut
Pengolahan Fitur Teks: Setiap fitur teks menjalani proses pre processing yang cermat (misalnya, penghapusan karakter tidak relevan, lowercase, penanganan multi-kata dengan underscore) untuk memastikan kualitas data yang optimal dan konsisten.
Vektorisasi TF-IDF: Untuk mengubah data teks menjadi format numerik yang dapat dianalisis oleh algoritma, digunakan Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer. TF-IDF efektif dalam menangkap pentingnya kata-kata dalam profil masing-masing film relatif terhadap seluruh koleksi film.
3. Perhitungan Kesamaan Konten
Metrik Kesamaan Kosinus: Kesamaan antar profil film yang telah diubah menjadi vektor numerik dihitung menggunakan Cosine Similarity. Metrik ini mengukur sudut antara dua vektor, di mana sudut yang lebih kecil menunjukkan kemiripan yang lebih tinggi. Hasilnya adalah matriks kesamaan yang menunjukkan seberapa mirip setiap film dengan setiap film lainnya.
4. Prioritisasi Rekomendasi Berdasarkan Kualitas
Metrik weighted_rating: Untuk memastikan rekomendasi yang diberikan tidak hanya relevan secara konten tetapi juga memiliki kualitas tinggi, metrik weighted_rating diimplementasikan. Metrik ini menggabungkan users_rating dengan votes untuk memberikan skor kualitas film yang lebih andal dan adil. 

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).



Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
