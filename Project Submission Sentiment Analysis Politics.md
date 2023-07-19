# Laporan Proyek Machine Learning -  _Tweet Sentiment Analysis_ Bernuansa Politik pada Pemilihan Gubernur DKI Jakarta 2017

#### Penulis: Billy Akbar Prabowo

 Proyek ini ditulis untuk pemenuhan _submission_ pertama pada _predictive analytics_. Proyek ini membahas mengenai model klasifikasi menggunakan metode _Naive Bayes_ pada data kumpulan _tweet_ bernuansa politik pada Pemilihan Gubernur DKI Jakarta 2017.

## Domain Proyek

<br>

![Twitter](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/3faf122f-ee94-4bff-bea8-7a97b4a90fbf)

**[Gambar 1: Ilustrasi Twitter](https://www.inews.id/finance/bisnis/tingkatkan-pendapatan-twitter-kembali-izinkan-iklan-politik)**

<br>

Indonesia sebagai negara demokrasi menjamin masyakaratnya untuk bebas berpendapat melalui Undang-Undang Dasar (UUD) 1945. Dengan perkembangan teknologi yang semakin pesat, salah satu _platform_ digital, Twitter, menjadi salah satu komponen utama dalam yang mampu menyebarluaskan informasi kepada masyarakat dan berperan  penting dalam melihat sentimen pada suatu topik[1,2]. Di sisi lain, pesta demokrasi akan digelar di Indonesia pada tahun 2024, bertepatan pada pemilihan anggota legislatif dan eksekutif di tingkat daerah maupun nasional. Hal tersebut menjadi penting untuk pengusaha untuk melakukan strategi bisnis yang sesuai supaya tetap memiliki _positive growth_ pada sentimen politik yang akan terjadi di tahun 2024. Maka dari itu, analisis sentimen _tweet_ berkaitan pada politik dapat dipelajari dari mengklasifikasikan dan memprediksi sentimen masyarakat menggunakan model algoritma yang sesuai. Proyek ini akan menganalisis kumpulan _tweet_ saat Pemilihan Gubernur DKI Jakarta tahun 2017 yang diklasifikasikan dengan menggunakan algoritma _Naive Bayes_ untuk mendapatkan akurasi optimal.

Referensi Utama Proyek: [THE IMPLEMENTATION OF THE MACHINE LEARNING ALGORITHM FOR THE SENTIMENT ANALYSIS OF INDONESIA’S 2019 PRESIDENTIAL ELECTION](https://journals.iium.edu.my/ejournal/index.php/iiumej/article/view/1532/790) 

## Business Understanding
Berdasarkan dari latar belakang tersebut, maka pada laporan ini akan mencakup beberapa aspek berikut, mencakup:

### Problem Statements
- Bagaimana cara melakukan pra-pemrosesan data agar dapat digunakan untuk melatih model dari _dataset_ yang tersedia?
- Bagaimana cara membuat model yang akan digunakan untuk memprediksi sentimen?
- Apakah model algoritma yang dibuat bisa dikatakan akurat?
  
### Goals
- Mampu mengetahui dan melakukan persiapan data untuk dapat dilatih oleh model.
- Mampu membuat model machine learning yang dapat mengklasifikasi sentimen masyarakat mengenai politik pada Pemilihan Gubernur DKI Jakarta 2017.
- Mampu membuat model algoritma yang digunakan dengan menghitung akurasi dan mengevaluasi model tersebut menggunakan _confusion matrix_.

### Solution statements
- Melakukan persiapan / preparasi data menggunakan _Term Frequency-Inverse Document Frequency (TF-IDF)_
- Membandingkan hasil akurasi dari model dengan _confusion matrix_

## Data Understanding
_Dataset_ yang digunakan dalam proyek ini merupakan data _tweet_ pada Pemilihan Gubernur DKI Jakarta 2017 yang dapat diunduh di  [Kaggle : Sentiment Analysis](https://www.kaggle.com/datasets/deniyulian/sentiment-analysis).

Informasi tentang _dataset_:

- Data berjudul _dataset_tweet_sentiment_pilkada_DKI_2017.csv_ dan memiliki format CSV (Comma-Seperated Values).
- Data memiliki 900 _rows_ yang berisikan mengenai _tweet_ dan 4 _column_ yang terdiri dari nomor, sentimen, pasangan calon, dan _text tweet_.
- Tidak ada _missing value_ dalam _dataset_.
  
### Variabel-variabel pada _dataset_:
Terdapat dua variabel utama yang digunakan pada proyek ini, yaitu sebagai berikut:
- sentiment : merupakan jenis sentimen dari masing-masing _tweet_, terdiri dari dua pilihan, yaitu _positive_ dan _negative_.
- tweet text : merupakan kumpulan _tweet_ yang berisikan mengenai topik politik saat pemilihan Gubernur DKI Jakarta pada 2017.


## _Data Preparation_
Sebelum memulai pengolahan data, maka sebelumnya diperlukan beberapa tahapan seperti _import library_ dan _import dataset_ ke dalam kode.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/fc89b49c-89be-4673-ae1b-bb083c349466)

**Gambar 2: Paket _library_ yang akan digunakan saat memodelkan data**

Setelah itu, hubungkan Google Colab dengan sumber _dataset_ (pada kasus ini _dataset_ disimpan pada Google Drive) dengan kode berikut,

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/3d8e33a1-79a0-4cee-88d1-262b147e49db)\
**Gambar 3: Kode yang digunakan untuk menghubungkan Google Colab pada sumber _dataset_**

Setelah dapat menghubungkan Google Colab dengan sumber _dataset_, maka selanjutnya adalah membaca _dataset_ dengan menuliskan kode dengan judul yang sesuai seperti di bawah ini,

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/3629f617-0c13-4e7e-9d0b-e96a67f69137)

**Gambar 4: Kode yang digunakan untuk membaca _dataset_**

Setelah _dataset_ terbaca, maka proses selanjutnya adalah menghilangkan tanda baca, tautan, maupun simbol yang tidak perlu (karakter selain pembentuk kata) dengan cara menuliskan kode di bawah ini. Hal ini bertujuan untuk mengurangi _noise_ pada model. 

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/5d1d8960-6e0a-4f51-b1b4-80074f75c927)

**Gambar 5: Proses menghilangkan tanda baca, tautan, maupun simbol yang tidak perlu pada  _dataset_**

Setelah berhasil menghilangkan simbol, tautan, dan tanda baca yang tidak perlu, maka kita dapat melakukan visualisasi data untuk mendapatkan gambaran secara menyeluruh mengenai jumlah data maupun _tagging_ sentimennya. Dalam kasus ini, visualisasi data dilakukan dengan _pie chart_ dan memberikan informasi bahwa dari _dataset_ terdapat 50% _tweet_ dengan sentimen negatif dan 50% _tweet_ dengan sentimen positif.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/c5b9dbd8-4d9c-4ed0-81f1-eea02a9cfa46)

**Gambar 6: Visualisasi  _dataset_ berdasarkan jumlah _tweet_ dan sentimennya**

## Modeling
Tahapan ini membahas mengenai model _machine learning_ yang digunakan untuk menyelesaikan permasalahan. Kita bisa membagi data uji dan data latih dengan data uji sebesar 20% untuk masing-masing tipe sentimen.
![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/1dcf692d-2d16-4f1e-987d-2bea13159fee)

**Gambar 7: Pembagian _dataset_ menjadi data uji dan data latih**

Setelah itu, kita bisa mengaplikasikan _Term Frequency-Inverse Document Frequency (TF-IDF)_ yang akan menilai dan melakukan tokenisasi dan digunakan untuk mengetahui frekuensi suatu kata muncul di dalam dokumen.
![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/57334ead-4d08-4e6d-a6f9-bebe4ebfe095)

**Gambar 8: Melakukan algoritma TF-IDF untuk mengetahui frekuensi suatu kata muncul di dalam dokumen**
Dengan catatan, walaupun TF-IDF memiliki kelebihan mudah dan efisien (tidak perlu melakukan tokenisasi), namun metode ini menghilangan informasi kategori pada tiap dokumen/_dataset_.

Setelah itu, kita dapat mengubah teks _tweet_ menjadi representasi dalam pelatihan data pada TF-IDF sebelum melakukan algoritma _Naive Bayes_.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/c16085ec-9ce8-4283-961f-a139f8cd45f1)

**Gambar 9: Mengubah teks _tweet_ menjadi representasi dalam pelatihan data pada TF-IDF**

Metode algoritma _Naive Bayes_ metode yang dapat memprediksi kelas/kategori probabilitas keanggotaan, seperti probabilitas bahwa sampel yang diberikan
milik kelas/kategori tertentu [3]. Metode ini didasarkan pada teorema Bayes yang mengasumsikan bahwa peluang dari 2 kejadian terjadi saling memengaruhi. Maka dari itu, metode _Naive Bayes_ mampu mengasumsikan probabilitas ketika kita sudah mengetahui probabilitas tertentu lainnya. Metode ini terkenal mudah dan sederhana. Di sisi lain, walaupun metode ini dapat mengasumsikan probabilitas ketika kita sudah dapat mengetahui probabilitas tertentu lainnya, jika probabilitas kondisionalnya nol maka prediksi akan bernilai nol juga.

Algoritma _Naive Bayes_ akan menghasilkan akurasi yang dapat dihitung dari jumlah data yang diklasifikan dengan benar dibagi dengan jumlah semua data yang diklasifikasikan. Untuk dapat menggunakan metode ini, maka kita perlu mengaktifkannya dengan melakukan _import library_ berikut,

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/b6a59e2f-5264-4100-b766-d0ac222181d5)

**Gambar 10: Melakukan _import library Multinomial Naive Bayes_**

Setelah itu, maka kita dapat melakukan pelatihan data yang telah dipreparasi di atas menggunakan algoritma _Naive Bayes_ untuk mendapatkan akurasinya. Pada kasus ini, akurasi didapatkan 75.56%.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/e5ba0dd6-f5e3-4d49-bcdc-721a3115e9d4)

**Gambar 11: Algoritma _Multinomial Naive Bayes_**

## Evaluation
Pada bagian ini, kita akan mencoba menguji hasil akurasi _Naive Bayes_ dengan metode algoritma _confusion matrix_. Dari akurasi yang diperoleh dengan metode _Naive Bayes_ (75.56%), maka masih ada kemungkinan salah klasifikasi pada _tweet_ karena adanya kemungkinan untuk _False Positive_ maupun _False Negative_. Maka dari itu, _confusion matrix_ dapat membantu untuk dapat mendapatkan pengukuran lain (yaitu presisi, _recall_, dan skor F1).

_Confusion matrix_ merupakan metode evaluasi model dalam melakukan klasifikasi yang terdiri dari ringkasan tabel jumlah perdiksi yang benar dan salah dengan 4 matriks nilai, yaitu _True Positive (TP), True Negative (TN), False Positive (FP)_, dan _False Negative (FN)_. Suatu model _confusion matrix_ dapat dikatakan bagus jika memiliki nilai _True Positive_ dan _True Negative_ yang tinggi.

![1_fxiTNIgOyvAombPJx5KGeA](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/f213e867-231e-49ba-8e66-a1fcd9f84341)

**[Gambar 12: Ilustrasi _Confusion Matrix_](https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826)**

Berikut adalah implementasi model _confusion matrix_ yang menghasilkan matriks 2x2 dan nilai presisi, _recall_, dan skor F1.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/e3135539-7c6c-4933-b7c3-80dc0dce372e)

**Gambar 13: Hasil analisis metode _confusion matrix_ yang menghasilkan matriks 2x2 dan nilai presisi, _recall_ dan skor F1**

Jika dilihat pada matriks 2x2, hasil yang diperoleh untuk TP dan TN memiliki nilai yang relatif tinggi. Hal ini membuktikan bahwa model ini sudah lumayan bagus untuk melakukan klasifikasi sentimen. Selain itu, didapatkan juga nilai presisi, _recall_, dan skor F1. Presisi adalah pembagian antara TP dengan (TP+FP) - atau perbandingan hasil yang positif secara benar dengan semua data yang dikategorikan sebagai positif. Nilai presisi pada model ini adalah 76.02%, atau sedikit lebih tinggi dengan akurasi _Naive Bayes_. Sedangkan _recall_ adalah pembagian antara TP dengan (TP+FN) atau pembagian hasil yang positif secara benar dengan penjumlahan semua hal yang seharusnya dikategorikan positif. Diperoleh nilai _recall_ adalah 75.56% atau sama dengan akurasi pada _Naive Bayes_. Sedangkan yang terakhir adalah skor F1. Skor ini diperoleh dari perkalian dari perbandingan (presisi x _recall_) dengan (presisi + _recall) yang hasilnya dikali 2. Dari pengujian _confusion matrix_ diperoleh F1 sebesar 75.45% yang hasilnya hampir mirip dengan _recall_ dan akurasi _Naive Bayes_.

## Kesimpulan

Dari percobaan ini maka dapat disimpulkan beberapa hal yaitu:
- Preparasi data dilakukan dengan cara menghilangkan tanda baca, simbol, dan tautan untuk menghindari _noise_ serta TF-IDF
- Pemodelan dilakukan dengan metode _Naive Bayes_ yang memperoleh akurasi 75.56% dan _confusion matrix_ yang memperoleh nilai presisi sebesar 76.02%, nilai _recall_ sebesar 75.56%, dan F1 sebesar 75.45%.
- Dapat disimpulkan bahwa kedua metode tersebut memiliki akurasi yang relatif tinggi dan hasil yang serupa.
  
## Referensi
- [1] Lawelai, H., Sadat, A., & Suherman, A. (2022). Democracy and Freedom of Opinion in Social Media: Sentiment Analysis on Twitter. PRAJA: Jurnal Ilmiah Pemerintahan, 10(1), 40-48.
- [2] Buntoro, G. A., Arifin, R., Syaifuddiin, G. N., Selamat, A., Krejcar, O., & Hamido, F. (2021). The implementation of the machine learning algorithm for the sentiment analysis of Indonesia’s 2019 presidential election. IIUM Engineering Journal, 22(1), 78-92.
- [3] Leung, K. M. (2007). Naive bayesian classifier. Polytechnic University Department of Computer Science/Finance and Risk Engineering, 2007, 123-156.

