# Laporan Proyek Machine Learning -  _Tweet Sentiment Analysis_ Bernuansa Politik pada Pemilihan Gubernur DKI Jakarta 2017

#### Penulis: Billy Akbar Prabowo

 Proyek ini ditulis untuk pemenuhan _submission_ pertama pada _predictive analytics_. Proyek ini membahas mengenai model klasifikasi menggunakan metode _Naive Bayes_ pada data kumpulan _tweet_ bernuansa politik pada Pemilihan Gubernur DKI Jakarta 2017.

## Domain Proyek

<br>

![Twitter](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/3faf122f-ee94-4bff-bea8-7a97b4a90fbf)

[Referensi gambar](https://www.inews.id/finance/bisnis/tingkatkan-pendapatan-twitter-kembali-izinkan-iklan-politik)

<br>

Indonesia sebagai negara demokrasi menjamin masyakaratnya untuk bebas berpendapat melalui Undang-Undang Dasar (UUD) 1945. Dengan perkembangan teknologi yang semakin pesat, salah satu _platform_ digital, Twitter, menjadi salah satu komponen utama dalam yang mampu menyebarluaskan informasi kepada masyarakat dan berperan  penting dalam melihat sentimen pada suatu topik[1,2]. Di sisi lain, pesta demokrasi akan digelar di Indonesia pada tahun 2024, bertepatan pada pemilihan anggota legislatif dan eksekutif di tingkat daerah maupun nasional. Hal tersebut menjadi penting untuk pengusaha untuk melakukan strategi bisnis yang sesuai supaya tetap memiliki _positive growth_ pada sentimen politik yang akan terjadi di tahun 2024. Maka dari itu, analisis sentimen _tweet_ berkaitan pada politik dapat dipelajari dari mengklasifikasikan dan memprediksi sentimen masyarakat menggunakan model algoritma yang sesuai. Proyek ini akan menganalisis kumpulan _tweet_ saat Pemilihan Gubernur DKI Jakarta tahun 2017 yang diklasifikasikan dengan menggunakan algoritma _Naive Bayes_ dan _Support Vector Machine/SVM_ untuk mendapatkan akurasi paling optimal.

Referensi Utama Proyek: [THE IMPLEMENTATION OF THE MACHINE LEARNING ALGORITHM FOR THE SENTIMENT ANALYSIS OF INDONESIA’S 2019 PRESIDENTIAL ELECTION](https://journals.iium.edu.my/ejournal/index.php/iiumej/article/view/1532/790) 

## Business Understanding
Berdasarkan dari latar belakang tersebut, maka pada laporan ini akan mencakup beberapa aspek berikut, mencakup:

### Problem Statements
- Bagaimana cara melakukan pra-pemrosesan data agar dapat digunakan untuk melatih model dari _dataset_ yang tersedia?
- Bagaimana cara membuat model yang akan digunakan untuk memprediksi?
- Bagaimana sentimen masyarakat yang direpresentasikan oleh kumpulan _tweets_ pada Pemilihan Gubernur DKI Jakarta 2017?
- Apakah model algoritma yang dibuat bisa dikatakan akurat?
  
### Goals
- Mampu mengetahui dan melakukan persiapan data untuk dapat dilatih oleh model.
- Mampu membuat model machine learning yang dapat memprediksi sentimen masyarakat mengenai politik pada Pemilihan Gubernur DKI Jakarta 2017.
- Mampu membuat model algoritma yang digunakan dengan menghitung akurasi dan mengevaluasi model tersebut menggunakan _confusion matrix_.

### Solution statements
- Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Referensi
[1] Lawelai, H., Sadat, A., & Suherman, A. (2022). Democracy and Freedom of Opinion in Social Media: Sentiment Analysis on Twitter. PRAJA: Jurnal Ilmiah Pemerintahan, 10(1), 40-48.
[2] Buntoro, G. A., Arifin, R., Syaifuddiin, G. N., Selamat, A., Krejcar, O., & Hamido, F. (2021). The implementation of the machine learning algorithm for the sentiment analysis of Indonesia’s 2019 presidential election. IIUM Engineering Journal, 22(1), 78-92.

