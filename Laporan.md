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

Data yang diambil adalah data IMDb Indonesian Movies dari kaggle. Adapun data yang diambil adalah Terdapat lebih dari 1200+ Film Indonesia dalam dataset yang terdiri dari 11 kolom berisi informasi terkait setiap film. Rincian data berjumlah 1.272 data dengan 11 variabel. 
URL : (https://www.kaggle.com/datasets/dionisiusdh/imdb-indonesian-movies)

Variabel-variabel pada dataset adalah sebagai berikut:
- 'title' : Judul lengkap dari film, ini identifikasi utama sebuah film.
- 'year' : Tahun rilis film, menunjukkan kapan film tersebut pertama kali dipublikasikan atau ditayangkan.
- 'description' : Ringkasan atau sinopsis cerita film, ini sangat penting untuk content-based filtering karena menjelaskan isi film secara detail.
- 'genre' : Kategori atau jenis film berdasarkan tema, gaya, atau alur cerita.
- 'rating' : Klasifikasi usia penonton atau tingkat kedewasaan konten film. Ini menunjukkan batasan usia minimum yang disarankan untuk menonton film tersebut berdasarkan isi (kekerasan, bahasa, tema dewasa, dll.).
- 'user_rating' : Rata-rata penilaian yang diberikan pengguna (bukan kritikus profesional).
- 'votes' : Jumlah total suara atau penilaian yang berkontribusi pada 'rating' atau 'users rating' sebuah film. Menunjukkan seberapa banyak orang yang telah memberikan penilaian. Semakin tinggi jumlah 'votes', semakin representatif 'rating' tersebut.
- 'languages' : Bahasa utama yang digunakan dalam film.
- 'directors' : Nama sutradara yang mengarahkan film. Sutradara adalah elemen penting dalam identifikasi gaya dan kualitas film.
- 'actors' : Nama-nama aktor/aktris utama yang berperan dalam film. Kehadiran aktor tertentu bisa menjadi daya tarik bagi penonton.
- 'runtime' : Durasi atau lama waktu film dalam satuan menit.

**Rubrik/Kriteria Tambahan**:
Berdasarkan EDA, dilakukan :
1. Cek tipe data : Jika ditinjau jauh, kolom 'votes' dan 'runtime' seharusnya berupa int karena votes adalah jumlah dan runtime ini waktu (terdapat imbuhan min sehingga menjadi object, perlu dihapus). Selain itu, untuk kategori lebih baik 'year' menjadi object dibanding int.
2. Cek data unique : Diperoleh insight:
- Jika 'title' merupakan identitas film seharusnya 1272 unique value semua tetapi ternyata hanya 1262. perlu dicek duplikasinya.
Jika 'year' memiliki 62 unique value kemungkinan besar tahun nya sangat jauh dari yang terbaru hingga yang terlama.
- Jika 'description' hanya memiliki 820 unique value, kemungkinan terdapat 432 nilai tanpa deskripsi. perlu dicek missing value karena ini penting jika memang tidak ada bisa diganti dengan string kosong ' '. Hal ini agar tidak mempengaruhi isi film.
- Jika 'genre' terdapat kolom berisi nilai nan, maka ini masuk ke dalam missing value sehingga perlu ditangani bisa diganti string kosong ' ' atau 'unknown genre'.
- Jika 'rating' terdapat kolom berisi nilai nan, maka ini masuk ke dalam missing value sehingga perlu ditangani bisa diganti string kosong ' ' atau 'unknown'.
- Jika 'actors' memiliki nilai unik 1266 perlu dicek untuk nilai lainnya apakah ada yang kosong atau memang ada yang sama. Selanjutnya, kolom ini kemungkinan besar berisi daftar aktor yang dipisahkan koma dalam format string (seperti terlihat di contoh : ["['Aktor1', 'Aktor2']"]). Oleh karena itu, perlu membersihkan format string ini (menghilangkan [], '', "), memisahkan nama-nama aktor, dan mengolahnya menjadi daftar individu untuk digabungkan ke profil konten.
- Jika 'runtime' terdapat missing value maka perlu ditangani bisa diganti dengan nilai median.
3. Cek statistik deskriptif : masih hanya 2 kolom dan belum mendapatkan insight yang berarti.
4. Cek missing value :
  - Untuk 'description' dan 'genre', penanganan NaN dengan string kosong adalah langkah yang tepat karena mereka akan digunakan untuk text vectorization. Bisa diganti dengan string kosong atau keterangan.
  - Untuk 'rating', bisa diganti dengan string kosong atau keterangan.
  - Untuk 'runtime' bisa diganti dengan nilai mediannya.
    
## Data Preparation
Adapun langkah pada data preparation yang dilakukan:
1. Mengubah tipe data yang benar : berdasarkan hasil data understanding, ada beberapa kolom yang perlu dicermati terkait data hal ini akan berpengaruh terutama jika dalam analisis numerik. Pada langkah ini, kolom 'votes' dan 'runtime' berubah menjadi int dengan khusus runtime kata 'min' dihilangkan. Selain itu, tahun diubah menjadi string agar melihat kategori film yang muncul di tahun tertentu.
2. Statistik deskriptif setelah pre-processing : dilakukan setelah tipe data numerik sudah benar, disini terdapat 'user_rating', 'votes', dan 'runtime'. Insight:
- Imputasi runtime: Pertimbangkan lagi untuk mengganti nilai 0 di runtime (yang berasal dari missing values) dengan median dari kolom runtime (sekitar 89 menit) atau nilai yang lebih realistis. Ini akan membuat fitur runtime lebih bermakna.
- Penanganan votes: Karena distribusi votes sangat skewed, saat menggunakan votes untuk mengurutkan atau menilai kualitas film, pertimbangkan untuk menggunakan log transformasi atau metrik seperti Weighted Rating untuk menstabilkan pengaruh film dengan votes sangat tinggi.
- Analisis Outlier: Mungkin ada baiknya melihat film-film dengan year yang sangat tua, users_rating sangat rendah/tinggi, atau votes sangat tinggi untuk memahami karakteristiknya lebih lanjut.
3. Metode Weight untuk Votes : Cara untuk memberikan nilai kualitas yang lebih adil dan kredibel pada sebuah film dibandingkan hanya melihat rata-rata rating mentahnya saja. Tujuannya adalah menghindari bias dari film-film yang memiliki rating sangat tinggi tetapi hanya dari sedikit votes.
Selain itu, juga memberikan bobot lebih pada rating film yang didukung oleh jumlah votes yang besar (lebih teruji). Skor kualitas film yang dihitung menggunakan formula 'Weighted Rating' (mirip dengan IMDb), yang mempertimbangkan users_rating dan jumlah votes untuk memberikan penilaian yang lebih kredibel dan stabil. Film dengan rating tinggi dari sedikit votes akan "ditarik" mendekati rata-rata global, sementara film dengan rating tinggi dari banyak votes akan mempertahankan ratingnya. Hal ini menambah kolom baru pada dataset.
4. Menangani missing value : karena berdasarkan data understanding terdapat beberapa kolom dengan nilai kosong, maka disini akan diubah:
- kolom 'description' yang missing value diganti dengan string kosong.
- kolom 'genre' yang missing value diganti dengan 'Unknown Genre'.
- kolom 'directors' yang missing value diganti dengan 'Unknown'.
- kolom 'rating' yang missing value diganti dengan 'Unknown'.
- kolom 'runtime' yang bernilai missing value dan nol diganti dengan nilai mediannya.
5. Menangani Duplikasi : duplikat seringkali ada dalam dataset, untuk film judul merupakan identitas yang seharusnya bernilai unique. Namun, ada beberapa film dengan judul yang sama dan tahun berbeda mungkin ini indikasi remake. Oleh karena itu, untuk menghindari salah paham judul film akan diberi identitas tahun disampingnya.
6. Menangani Kolom 'actors' : Jika ditilik kolom ini memiliki bentuk ['Aktor A','Aktor B','dst'] yang mana akan sulit jika dilanjutkan dalam analisis sehingga bertujuan untuk membersihkan dan menormalisasi kolom teks yang berisi banyak nilai dalam satu sel (seperti actors, directors, genre) agar setiap elemen menjadi token yang bersih, lowercase, dan multi-kata dihubungkan dengan underscore, serta disimpan dalam format list Python.
7. Univariate Analysis : bertujuan untuk melihat lebih jauh visual dari masing-masing kolom. Dalam hal ini dibagi menjadi kolom numerik dan kategorikal. Berdasarkan kolom numerik, nilai dari 'user_rating','runtime','weight_rating' cenderung berdistribusi normal.  Namun untuk kolom 'votes' histogramnya adalah yang paling mencolok karena menunjukkan distribusi yang sangat ekstrem (highly skewed). Mayoritas film memiliki jumlah votes yang sangat sedikit (terkonsentrasi di dekat 0), sementara hanya segelintir film yang memiliki votes puluhan hingga ratusan ribu. Pada data kategorikal, Visualisasi sudah cukup jelas mulai dari 'year' terbanyak 2019, 'rating' terbanyak tidak diketahui, 'languages' terbanyak Indonesian. Namun untuk 'genre','directions' dan actors'diwakili dengan abjad namun dapat menunjukkan visual terbanyak hingga yang sedikit. N
8. Multivariate Analysis : hal ini dilakukan untuk melihat hubungan antar variabel dengan yang lain. Namun dalam hal ini dilakukan analisis dengan data numerik, hubungan terkuat hanya terjadi pada 'users_rating' dengan 'weighted_rating'. Sebenarnya, banyak sekali outlier yang berada pada kolom numerik tetapi hal in bisa diabaikan karena nilai outlier yang rendah pada weighted_rating sebenarnya merepresentasikan film-film yang, bahkan setelah diboboti berdasarkan jumlah votes, masih dianggap memiliki kualitas yang sangat rendah oleh pengguna. Ini adalah informasi yang valid dan penting untuk tujuan ranking.
9. TF-IDF Vectorizer atau Term Frequency-Inverse Document Frequency berdasarkan genre: teknik statistik yang digunakan untuk mengubah teks menjadi representasi numerik yang menangkap seberapa penting sebuah kata dalam sebuah dokumen berisi film yang relatif terhadap seluruh koleksi dataset. Tujuannya adalah mengkonversi data teks yang tidak terstruktur seperti genre menjadi matriks numerik yang dapat dipahami oleh algoritma Machine Learning. Cara kerjanya adalah TfidfVectorizer menghasilkan sebuah matriks sparse (banyak nilai nol) di mana setiap baris merepresentasikan satu film, dan setiap kolom merepresentasikan sebuah kata unik dari seluruh korpus. Nilai dalam sel adalah bobot TF-IDF dari kata tersebut untuk film yang bersangkutan.
10. TD-IDF berdasarkan actors : penjelasannya hampir sama dengan nomor 9. hanya lebih spesifik bahwa tujuannya adalah mengkonversi data teks yang tidak terstruktur seperti actors menjadi matriks numerik yang dapat dipahami oleh algoritma Machine Learning.

## Modeling

Tahapan ini membahas mengenai model sisten rekomendasi yang dibuat, adapun yang dipiih adalah Content-based Filtering. Dalam hal ini dilakukan modelling berdasarkan genre dan actors. Berikut penjelasannya :
1. Cosine Similarity : Setelah teks dikonversi menjadi vektor numerik menggunakan TF-IDF, diperlukan cara untuk mengukur seberapa mirip vektor-vektor ini. Tujuan teknik ini adalah mengukur kesamaan antara dua dokumen (film) berdasarkan representasi vektor TF-IDF mereka. Perhitungan Cosine Similarity menghasilkan matriks kesamaan di mana setiap sel [i, j] berisi skor kesamaan antara film i dan film j. Matriks ini adalah dasar untuk menemukan film-film yang paling mirip.
2. Sajikan top-N recommendation sebagai output.
- Top-5 recommendation judul film berdasarkan genre:
  ![image](https://github.com/user-attachments/assets/63a4a213-a9c7-43df-9f7b-b888cdeae754)
- Top-5 recommendation judul film berdasarkan actors:
<img width="434" alt="image" src="https://github.com/user-attachments/assets/783827b2-d9c8-4233-b03c-fbb19eb47df3" />


## Evaluation
Karena sistem rekomendasi content-based ini berfokus pada kesamaan fitur maka metrik evaluasi akan lebih berorientasi pada kualitas rekomendasi dan relevansi. Metrik yang digunakan adalah Top-N Accuracy dimana mengukur proporsi rekomendasi yang relevan di antara N rekomendasi teratas yang diberikan. Namun, ini dapat diukur secara kuantitatif dengan Precision@K.
Precision@K adalah metrik standar dan kuantitatif yang mengukur "Top-N Accuracy" tersebut. Adapun rumusnya:
![image](https://github.com/user-attachments/assets/486c5067-4c5d-4fff-9101-ea65cafa79c7)
Precision@K sangat baik untuk mengukur seberapa "bersih" daftar rekomendasi. Ini penting karena pengguna biasanya hanya melihat beberapa rekomendasi teratas. Agar memastikan sebagian besar dari yang mereka lihat adalah relevan.

Adapun hasil darai Precision@K:
1. Berdasarkan genre
Precision@5 (berbasis genre) = 1.00:
Ini berarti 100% (5 dari 5) film yang direkomendasikan dalam Top-5 memiliki genre yang sama dengan film input ('Milea (2020)').
Artinya: Jika relevansi didefinisikan sebagai "berbagi genre utama yang sama", maka sistem rekomendasi berbasis genre ini bekerja dengan sangat baik untuk film 'Milea (2020)', menghasilkan rekomendasi yang sangat relevan berdasarkan genre.
2. Berdasarkan actors
Precision@5 (berbasis AKTOR) = 1.00:
Ini adalah skor yang sempurna! Artinya, 100% (5 dari 5) film yang direkomendasikan dalam Top-5 ini dianggap "relevan" berdasarkan kriteria kesamaan aktor yang baru Anda definisikan (yaitu, memiliki setidaknya satu aktor yang sama dengan film input 'Milea (2020)').
Ini menunjukkan bahwa sistem rekomendasi berbasis aktor Anda, ketika dievaluasi dengan kriteria yang sesuai, sangat efektif dalam menemukan film-film dengan aktor yang relevan untuk film input 'Milea (2020)'.

Tujuan utama dari pengembangan model ini adalah untuk menghasilkan rekomendasi film yang relevan, beragam, dan menarik bagi pengguna, sehingga meningkatkan pengalaman mereka dalam menemukan film.

**Conclusion** 
Berdasarkan problem statement dan tujuan, diperoleh hasil dari Machine Learning menggunakan TF-IDF Vectorizer dan Cosine Similarity bahwa 
1. Dari hasil yang diperoleh sebagai rekomendasi film berdasarkan genre dan actors dapat mempercepat dan mempersonalisasi pencarian.
2. Selain itu, dengan adanya 'weighted_rating' sebagai bobot dari 'users_Rating' dan 'votes' dapat meningkatkan penemuan konten relevan untuk memastikan rekomendasi yang diberikan adalah film-film berkualitas tinggi dan sesuai dengan minat pengguna.
3. Hasil dari evaluasi metrik Precision@K menunjukkan hasil 1.00 pada kedua basis yaitu genre dan actors sehingga hal ini sangat bagus untuk sistem rekomendasi.
