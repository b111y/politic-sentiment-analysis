# Laporan Proyek Machine Learning -  _Tweet Sentiment Analysis_ Bernuansa Politik pada Pemilihan Gubernur DKI Jakarta 2017

#### Penulis: Billy Akbar Prabowo

 Proyek ini ditulis untuk pemenuhan _submission_ pertama pada _predictive analytics_. Proyek ini membahas mengenai model klasifikasi menggunakan metode _Naive Bayes_ dan _confusion matrix_ pada data kumpulan _tweet_ bernuansa politik pada Pemilihan Gubernur DKI Jakarta 2017.

## Domain Proyek

<br>

![Twitter](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/3faf122f-ee94-4bff-bea8-7a97b4a90fbf)

**[Gambar 1: Ilustrasi Twitter](https://www.inews.id/finance/bisnis/tingkatkan-pendapatan-twitter-kembali-izinkan-iklan-politik)**

<br>

Indonesia sebagai negara demokrasi menjamin masyakaratnya untuk bebas berpendapat melalui Undang-Undang Dasar (UUD) 1945. Dengan perkembangan teknologi yang semakin pesat, salah satu _platform_ digital, Twitter, menjadi salah satu komponen utama dalam yang mampu menyebarluaskan informasi kepada masyarakat dan berperan  penting dalam melihat sentimen pada suatu topik[1,2]. Di sisi lain, pesta demokrasi akan digelar di Indonesia pada tahun 2024, bertepatan pada pemilihan anggota legislatif dan eksekutif di tingkat daerah maupun nasional. Hal tersebut menjadi penting untuk perusahaan atau sebuah organisasi untuk melakukan strategi mengenai keberlangsungan kegiatan dalam mengantisipasi potensi adanya gesekan sosial atau tidak di 2024. Maka dari itu, analisis sentimen _tweet_ berkaitan pada politik dapat dipelajari dari mengklasifikasikan dan memprediksi sentimen masyarakat menggunakan model algoritma yang sesuai. Proyek ini akan menganalisis kumpulan _tweet_ saat Pemilihan Gubernur DKI Jakarta tahun 2017 yang diklasifikasikan dengan menggunakan algoritma _Naive Bayes_ dan _confusion matrix_ untuk membandingkan hasil akurasi maupun prediksi tiap model.


## Business Understanding
Berdasarkan dari latar belakang tersebut, maka pada laporan ini akan mencakup beberapa aspek berikut, mencakup:

### Problem Statements
- Bagaimana hasil pemodelan dari algoritma  _Naive Bayes_?
- Bagaimana hasil pemodelan dari algoritma _confusion matrix_?
- Apakah model algoritma yang dibuat bisa dikatakan akurat?
  
### Goals
- Mampu menghitung akurasi dari pemodelan dari algoritma  _Naive Bayes_.
- Mampu menghitung presisi, _recall_ dan skor F1 dari pemodelan dari algoritma  _confusion matrix_.
- Mampu membandingkan hasil dari model algoritma _Naive Bayes_ _confusion matrix_.

### Solution statements
- Membandingkan hasil akurasi dari model _Naive Bayes_ dengan _confusion matrix_

## Data Understanding
_Dataset_ yang digunakan dalam proyek ini merupakan data _tweet_ pada Pemilihan Gubernur DKI Jakarta 2017 yang dapat diunduh di  [Kaggle : Sentiment Analysis](https://www.kaggle.com/datasets/deniyulian/sentiment-analysis).

**Tabel 1: Informasi mengenai _dataset_**
| Jenis                  | Keterangan                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Sumber                 |  [Kaggle : Sentiment Analysis](https://www.kaggle.com/datasets/deniyulian/sentiment-analysis)                     |
| Linsensi               |  CC0: Public Domain                                                                                               |
| Kategori               | Sosial                                                                                                            |
| Jenis & Ukuran berkas  | CSV (122KB)                                                                                                       |  
| Nama dataset           | dataset_tweet_sentiment_pilkada_DKI_2017.csv                                                                      |  
| Jumlah data            | 4 buah kolom (nomor, sentimen, pasangan calon, dan teks tweets), 900 baris                                        |  
  
### Variabel-variabel pada _dataset_:
Terdapat dua variabel utama yang digunakan pada proyek ini, yaitu sebagai berikut:

**Tabel 2: Informasi mengenai variabel**
| Variabel                  | Keterangan                                                                                                        |
| --------------------------| ----------------------------------------------------------------------------------------------------------------- |
| sentiment                 | jenis sentimen dari masing-masing _tweet_, terdiri dari dua pilihan, yaitu _positive_ dan _negative_.             |
| tweet text                | kumpulan _tweet_ yang berisikan mengenai topik politik saat pemilihan Gubernur DKI Jakarta pada 2017              |



## _Data Preparation_
Sebelum memulai pengolahan data, maka sebelumnya diperlukan beberapa tahapan seperti berikut:
- Mengimpor library dengan kode `import`
- Menghubungkan Google Colab dihubungkan dengan sumber _dataset_ (pada kasus ini _dataset_ disimpan pada Google Drive) dengan kode `import()`
- Membaca _dataset_ dengan kode `pd.read_csv()`
- Menghilangkan tanda baca, tautan, maupun simbol yang tidak perlu (karakter selain pembentuk kata) dengan tujuan untuk mengurangi _noise_ pada model
- Melakukan pembagian _dataset_ menjadi dua yaitu data uji dan data latih dengan data uji sebesar 20% untuk masing-masing tipe sentimen
- Mengaplikasikan _Term Frequency-Inverse Document Frequency (TF-IDF)_ `TfidfVectorizer()` yang akan menilai dan melakukan tokenisasi dan digunakan untuk mengetahui frekuensi suatu kata muncul di dalam dokumen




## Modeling
Pada tahap _modeling_, data yang telah dipreparasi akan diuji dengan metode _Naive Bayes_ dan metode _confusion matrix_. Metode algoritma _Naive Bayes_ metode yang dapat memprediksi kelas/kategori probabilitas keanggotaan, seperti probabilitas bahwa sampel yang diberikan milik kelas/kategori tertentu [3]. Metode ini didasarkan pada teorema Bayes yang mengasumsikan bahwa peluang dari 2 kejadian terjadi saling memengaruhi. Maka dari itu, metode _Naive Bayes_ mampu mengasumsikan probabilitas ketika user sudah mengetahui probabilitas tertentu lainnya. Metode ini terkenal mudah dan sederhana. Di sisi lain, walaupun metode ini dapat mengasumsikan probabilitas ketika user sudah dapat mengetahui probabilitas tertentu lainnya, jika probabilitas kondisionalnya nol maka prediksi akan bernilai nol juga. Algoritma _Naive Bayes_ akan menghasilkan akurasi yang dapat dihitung dari jumlah data yang diklasifikan dengan benar dibagi dengan jumlah semua data yang diklasifikasikan atau dapat ditulis pada rumus berikut:
<br>
$\text{Accuracy} = \frac{correctly-classified-items}{all-classified-items}$
<br>

_Confusion matrix_ merupakan metode evaluasi model dalam melakukan klasifikasi yang terdiri dari ringkasan tabel jumlah perdiksi yang benar dan salah dengan 4 matriks nilai, yaitu _True Positive (TP), True Negative (TN), False Positive (FP)_, dan _False Negative (FN)_. Suatu model _confusion matrix_ dapat dikatakan bagus jika memiliki nilai _True Positive_ dan _True Negative_ yang tinggi.

![1_fxiTNIgOyvAombPJx5KGeA](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/f213e867-231e-49ba-8e66-a1fcd9f84341)

**[Gambar 2: Ilustrasi _Confusion Matrix_](https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826)**

Terdapat beberapa aspek parameter untuk _confusion matrix_, yaitu presisi, _recall_, dan F1. Presisi adalah pembagian antara TP dengan (TP+FP) atau perbandingan hasil yang positif secara benar dengan semua data yang dikategorikan sebagai positif, atau dapat ditulis pada rumus berikut:
<br>
$\text{Presisi} = \frac{TP}{TP+FP}$
<br>

Sedangkan _recall_ adalah pembagian antara TP dengan (TP+FN) atau pembagian hasil yang positif secara benar dengan penjumlahan semua hal yang seharusnya dikategorikan positif, atau dapat diekspresikan dengan rumus berikut:
<br>
$\text{Recall} = \frac{TP}{TP+FN}$
<br>


Terakhir, skor F1 diperoleh dari perkalian dari perbandingan (presisi x _recall_) dengan (presisi + _recall) yang hasilnya dikali 2, atau diekspresikan pada rumus berikut:
<br>
$\text{F1} = \frac{2 \* Precision \* Recall}{Precision+Recall}$    
<br>

Dari aspek-aspek di atas, maka masing-masing nilainya akan dibandingkan untuk mengukur ketepatan antarmodel. Jika nilai mendekati 100% maka dapat dikatakan model tersebut semakin baik.

## Evaluation
Percobaan menggunakan model _Naive Bayes_, nilai akurasi diukur dengan rumus pada bab Modeling. Setelah dijalankan, maka didapatkan akurasi sebesar 75.56%. Dari hasil ini dapat diindikasikan bahwa ada kemungkinan salah klasifikasi pada _tweet_ karena adanya kemungkinan untuk _False Positive_ maupun _False Negative_. Maka dari itu, _confusion matrix_ dapat membantu untuk evaluasi model dan melihat nilai pengukuran lain (yaitu presisi, _recall_, dan skor F1).

Implementasi model _confusion matrix_ yang menghasilkan matriks 2x2, sebagai berikut dan nilai presisi, _recall_, dan skor F1.

![image](https://github.com/b111y/politic-sentiment-analysis/assets/84972036/e3135539-7c6c-4933-b7c3-80dc0dce372e)

**Gambar 3: Hasil analisis metode _confusion matrix_ yang menghasilkan matriks 2x2 dan nilai presisi, _recall_ dan skor F1**

Jika dilihat pada matriks 2x2, hasil yang diperoleh untuk TP dan TN memiliki nilai yang relatif tinggi. Hal ini membuktikan bahwa model ini sudah lumayan bagus untuk melakukan klasifikasi sentimen. Selain itu, didapatkan juga nilai presisi, _recall_, dan skor F1. Nilai presisi pada model ini adalah 76.02%, atau sedikit lebih tinggi dengan akurasi _Naive Bayes_. Sedangkan diperoleh nilai _recall_ adalah 75.56% atau sama dengan akurasi pada _Naive Bayes_. Terakhir,diperoleh F1 sebesar 75.45% yang hasilnya hampir mirip dengan _recall_ dan akurasi _Naive Bayes_.

Dari percobaan ini maka dapat disimpulkan beberapa hal yaitu:
- Preparasi data dilakukan dengan cara menghilangkan tanda baca, simbol, dan tautan untuk menghindari _noise_ serta TF-IDF
- Pemodelan dilakukan dengan metode _Naive Bayes_ yang memperoleh akurasi 75.56% dan _confusion matrix_ yang memperoleh nilai presisi sebesar 76.02%, nilai _recall_ sebesar 75.56%, dan F1 sebesar 75.45%.
- Dapat disimpulkan bahwa kedua metode tersebut memiliki akurasi yang relatif tinggi dan hasil yang serupa.
  
## Referensi
- [1] [Lawelai, H., Sadat, A., & Suherman, A. (2022). Democracy and Freedom of Opinion in Social Media: Sentiment Analysis on Twitter. PRAJA: Jurnal Ilmiah Pemerintahan, 10(1), 40-48.](https://jurnal.umsrappang.ac.id/praja/article/view/585)
- [2] [Buntoro, G. A., Arifin, R., Syaifuddiin, G. N., Selamat, A., Krejcar, O., & Hamido, F. (2021). The implementation of the machine learning algorithm for the sentiment analysis of Indonesia’s 2019 presidential election. IIUM Engineering Journal, 22(1), 78-92.](https://journals.iium.edu.my/ejournal/index.php/iiumej/article/view/1532)
- [3] [Leung, K. M. (2007). Naive bayesian classifier. Polytechnic University Department of Computer Science/Finance and Risk Engineering, 2007, 123-156.](https://cse.engineering.nyu.edu/~mleung/FRE7851/f07/naiveBayesianClassifier.pdf)

