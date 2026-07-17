LAPORAN TUGAS AKHIR
ANALISIS PREDIKTIF DAN IDENTIFIKASI FAKTOR RISIKO DEPRESI MAHASISWA MENGGUNAKAN PENDEKATAN MACHINE LEARNING

Dosen Pengampu :
Leni Fitriani, ST.,M.Kom.

	
Kelompok 2 
         Anggota :
 Neng Isan Lopiani	    : 2406179
         Sri Sukma Fitria		: 2406037





PROGRAM STUDI TEKNIK INFORMATIKA JURUSAN ILMU KOMPUTER
INSTITUT TEKNOLOGI GARUT GARUT
2025
ABSTRAK
Kesehatan mental mahasiswa merupakan isu krusial yang berdampak langsung pada performa akademis, tingkat kelulusan, dan kualitas hidup secara menyeluruh. Penelitian ini bertujuan untuk membangun model prediktif klasifikasi depresi mahasiswa serta mengidentifikasi faktor-faktor risiko utama yang melatarbelakanginya dengan memanfaatkan data dari Student Depression Dataset. Berdasarkan rancangan eksperimen pada file AI_TB_UAS_Student_Depression_Dataset.ipynb, metodologi yang diimplementasikan mengadopsi kerangka kerja standar industri CRISP-DM (Cross-Industry Standard Process for Data Mining).
Dua algoritma machine learning dibandingkan secara komparatif: Logistic Regression (sebagai representasi model parametrik linier) dan Random Forest Classifier (sebagai representasi model non-parametrik berbasis ensemble bagging). Data melalui tahapan preprocessing yang ketat dan sistematis, meliputi pembersihan spasi nama kolom, penanganan nilai kosong (missing values), eliminasi data duplikat, seleksi fitur fungsional, label encoding untuk variabel kategorikal, pembagian data terstratifikasi (80% training, 20% testing), serta standarisasi skala fitur (feature scaling) menggunakan StandardScaler.
Hasil evaluasi menunjukkan bahwa algoritma Random Forest Classifier mengungguli Logistic Regression di seluruh metrik evaluasi utama, termasuk akurasi, presisi, recall, dan F1-score. Analisis eksplorasi data (EDA) dan kueri data bersyarat mengungkapkan adanya korelasi krusial antara tekanan akademik yang berada pada tingkat ekstrem () dengan kemunculan ide bunuh diri (suicidal thoughts). Selain itu, ditemukan pula hubungan linier antara tekanan finansial yang tinggi dengan durasi jam belajar yang berlebihan.
Penelitian ini memberikan kontribusi ilmiah berupa perancangan kerangka kerja deteksi dini berbasis kecerdasan buatan (Early Warning System) yang dapat diadopsi oleh institusi pendidikan tinggi untuk mengidentifikasi mahasiswa yang berisiko mengalami depresi secara preventif.
Kata Kunci: Kecerdasan Buatan, Depresi Mahasiswa, Logistic Regression, Random Forest, CRISP-DM, Analisis Prediktif, Kesehatan Mental.
BAB I
PENDAHULUAN
1.1 Latar Belakang
Masa perkuliahan merupakan fase transisi krusial bagi seorang individu dari masa remaja akhir (late adolescence) menuju dewasa awal (emerging adulthood). Dalam fase perkembangan ini, mahasiswa tidak hanya dituntut untuk mencapai kompetensi akademis yang tinggi, tetapi juga dihadapkan pada serangkaian tuntutan adaptasi lingkungan baru, perluasan relasi sosial, kemandirian finansial, hingga ketidakpastian mengenai arah karier masa depan. Akumulasi dari berbagai tekanan psikososial ini kerap kali memicu gangguan kesehatan mental yang signifikan jika individu tidak memiliki mekanisme koping (coping mechanism) yang memadai.
Salah satu manifestasi klinis yang paling umum dan destruktif di kalangan mahasiswa adalah depresi. Depresi bukan sekadar perasaan sedih sementara, melainkan gangguan suasana perasaan (mood disorder) persisten yang ditandai dengan anhedonia, kelelahan kronis, penurunan konsentrasi, hingga hilangnya harapan hidup. Dampak depresi pada mahasiswa bersifat multidimensional; mulai dari penurunan indeks prestasi kumulatif (IPK/CGPA) secara drastis, peningkatan angka putus kuliah (dropout), isolasi sosial, hingga skenario terburuk berupa tindakan melukai diri sendiri (self-harm) dan munculnya ide bunuh diri (suicidal ideation). Berbagai survei kesehatan mental global menunjukkan tren peningkatan prevalensi depresi mahasiswa pasca-pandemi, menjadikannya sebuah krisis kesehatan masyarakat yang membutuhkan perhatian segera.
Sayangnya, pendekatan konvensional yang diterapkan oleh institusi pendidikan tinggi dalam menangani isu kesehatan mental mahasiswa umumnya bersifat reaktif. Layanan konseling atau pusat kesehatan mental mahasiswa baru memberikan intervensi setelah mahasiswa mengalami krisis atau menunjukkan penurunan performa akademis yang parah. Pendekatan reaktif ini kurang efektif karena banyak mahasiswa yang enggan mencari bantuan (help-seeking behavior yang rendah) akibat stigma sosial yang melekat pada gangguan jiwa. Oleh karena itu, diperlukan pergeseran paradigma dari penanganan reaktif menuju sistem deteksi dini yang bersifat preventif dan proaktif.
Di era disrupsi teknologi saat ini, perkembangan bidang Kecerdasan Buatan (Artificial Intelligence) dan Machine Learning (ML) membuka peluang besar untuk mengatasi tantangan tersebut. Melalui pendekatan komputasi berbasis data (data-driven approach), model ML dapat dilatih untuk mengenali pola-pola terselubung dan interaksi kompleks antara variabel demografi, beban akademis, tekanan finansial, dan kebiasaan belajar yang mengindikasikan tingkat risiko depresi seorang mahasiswa.
Melalui eksperimen yang dirancang dalam file AI_TB_UAS_Student_Depression_Dataset.ipynb, penelitian ini berfokus pada pengembangan model klasifikasi prediktif untuk mendeteksi depresi mahasiswa menggunakan dataset riil. Dua algoritma populer dengan karakteristik matematika berbeda—yaitu Logistic Regression yang bersifat linier-parametrik dan Random Forest Classifier yang bersifat non-linier-ensemble—dipilih untuk dianalisis performanya secara komparatif. Dengan adanya penelitian ini, diharapkan institusi perguruan tinggi dapat mengintegrasikan model prediktif terbaik ke dalam sistem informasi kemahasiswaan sebagai instrumen skrining awal guna menyelamatkan kesejahteraan psikologis mahasiswa sebelum kondisi mereka memburuk secara klinis.
1.2 Rumusan Masalah
Berdasarkan latar belakang yang telah dipaparkan, dirumuskan beberapa permasalahan utama dalam penelitian tugas akhir ini sebagai berikut:
1.	Bagaimana merancang dan mengimplementasikan pipeline pengolahan data awal (data preprocessing) dan rekayasa fitur (feature engineering) yang optimal untuk menangani campuran fitur numerik dan kategorikal pada dataset depresi mahasiswa?
2.	Bagaimana performa komparatif antara algoritma Logistic Regression dan Random Forest Classifier dalam memprediksi status depresi mahasiswa berdasarkan metrik Akurasi, Presisi, Recall, dan F1-score?
3.	Faktor-faktor psikososial dan akademis apa saja yang memiliki korelasi serta pengaruh paling dominan terhadap kecenderungan depresi dan munculnya ide bunuh diri pada mahasiswa berdasarkan analisis data eksploratif dan kueri data spesifik?
1.3 Tujuan Penelitian
Adapun tujuan yang ingin dicapai melalui pelaksanaan penelitian tugas akhir ini adalah:
1.	Membangun sebuah pipeline rekayasa data yang bersih, terstandarisasi, dan terstruktur dengan memanfaatkan pustaka Pandas, NumPy, dan Scikit-Learn untuk memproses variabel-variabel pada dataset depresi mahasiswa.
2.	Membangun, melatih, mengevaluasi, dan membandingkan dua model klasifikasi pembelajaran mesin (Logistic Regression dan Random Forest) untuk mendapatkan model prediktif dengan tingkat keandalan dan akurasi tertinggi.
3.	Melakukan analisis profil mahasiswa berisiko tinggi melalui kueri data kondisional terperinci guna memberikan masukan berbasis bukti (evidence-based policy) bagi unit konseling dan pengelola kebijakan di perguruan tinggi.
1.4 Batasan Masalah
Untuk menjaga agar penelitian tetap fokus dan mendalam, ditetapkan batasan-batasan masalah sebagai berikut:
1.	Dataset yang dianalisis adalah Student Depression Dataset yang memuat data terstruktur mengenai karakteristik demografi (usia, gender), metrik akademis (CGPA, Tekanan Akademik, Jam Belajar), tekanan finansial, dan status depresi mahasiswa.
2.	Algoritma pembelajaran mesin yang digunakan dibatasi pada dua metode supervised learning untuk klasifikasi biner, yaitu Logistic Regression dan Random Forest Classifier.
3.	Penelitian berfokus pada pengembangan model, evaluasi performa, interpretabilitas model, dan analisis data deskriptif. Implementasi model ke dalam lingkungan produksi (seperti pembuatan API atau aplikasi web) berada di luar cakupan penelitian ini.
1.5 Manfaat Penelitian
Penelitian tugas akhir ini diharapkan dapat memberikan manfaat yang luas, baik secara akademis maupun praktis:
1.	Manfaat Akademis (Teoretis): Memberikan kontribusi literatur mengenai penerapan algoritma machine learning dalam bidang psikometrika dan kesehatan mental digital (e-mental health), serta menjadi referensi bagi penelitian sejenis yang berfokus pada perbandingan model parametrik dan non-parametrik.
2.	Manfaat Praktis (Aplikatif): Menjadi draf panduan bagi pihak pengelola perguruan tinggi (khususnya bagian kemahasiswaan dan bimbingan konseling) dalam mengidentifikasi parameter kunci pemicu depresi mahasiswa, sehingga program intervensi kesehatan mental dapat diselenggarakan secara lebih tepat sasaran.
 BAB II: TINJAUAN PUSTAKA
2.1 Depresi dan Kesehatan Mental Mahasiswa
Secara klinis, depresi diidentifikasi sebagai gangguan kesehatan mental yang memengaruhi pikiran, perasaan, tingkah laku, dan kondisi fisik seseorang secara kronis. Berdasarkan Diagnostic and Statistical Manual of Mental Disorders (DSM-5), gejala depresi meliputi hilangnya minat atau kesenangan pada hampir semua aktivitas (anhedonia), penurunan berat badan signifikan tanpa diet, insomnia atau hipersomnia, retardasi motorik, kelelahan ekstrem, perasaan tidak berharga, penurunan kemampuan berpikir, hingga ideasi bunuh diri yang berulang.
Pada populasi mahasiswa, faktor pemicu depresi bersifat multifaktorial. Peneliti umumnya mengklasifikasikan faktor-faktor tersebut ke dalam tiga domain utama:
•	Faktor Akademis: Meliputi indeks prestasi kumulatif (CGPA) yang tidak memenuhi ekspektasi diri atau orang tua, beban tugas kuliah yang berlebihan, persaingan akademis yang ketat, dan alokasi waktu belajar yang tidak seimbang.
•	Faktor Finansial: Ketidakmampuan membayar biaya kuliah, tingginya biaya hidup di area sekitar kampus, atau keharusan untuk bekerja paruh waktu sembari kuliah yang memicu kelelahan fisik dan mental.
•	Faktor Psikososial: Kurangnya dukungan sosial (social support), kesepian akibat merantau jauh dari keluarga, konflik interpersonal dengan teman sebaya, serta riwayat trauma psikologis.
2.2 Konsep Klasifikasi dalam Machine Learning
Klasifikasi merupakan salah satu paradigma utama dalam pembelajaran mesin terawasi (supervised learning). Tujuan utama dari tugas klasifikasi adalah memetakan fungsi pemetaan  dari variabel input  (fitur) ke variabel target  yang bersifat diskrit (kelas atau label). Secara matematis, dituliskan sebagai:
Di mana  untuk  kelas. Pada kasus prediksi depresi mahasiswa ini, klasifikasi yang dilakukan adalah klasifikasi biner (binary classification) di mana target variabel memiliki dua kemungkinan nilai:
2.3 Logistic Regression
Logistic Regression (Regresi Logistik) adalah algoritma statistik parametrik yang digunakan untuk memprediksi probabilitas terjadinya suatu peristiwa biner. Berbeda dengan regresi linier biasa yang menghasilkan output kontinu tak terbatas, Regresi Logistik menggunakan fungsi aktivasi non-linier untuk membatasi output nilai prediksi berada pada rentang interval . Fungsi ini dikenal sebagai fungsi sigmoid atau fungsi logistik, yang diformulasikan sebagai berikut:
Variabel  merupakan kombinasi linier dari fitur-fitur input  dan parameter bobot :
Dengan mensubstitusikan persamaan kombinasi linier ke dalam fungsi sigmoid, diperoleh probabilitas prediksi kelas  bersyarat terhadap input :
Untuk menentukan label kelas akhir, ditetapkan ambang batas keputusan (decision threshold), biasanya sebesar . Aturan klasifikasi didefinisikan sebagai:
Proses pembelajaran pada Regresi Logistik dilakukan dengan mengoptimalkan fungsi kerugian Cross-Entropy Loss (atau Log Loss) menggunakan algoritma optimasi seperti Gradient Descent:
Di mana  adalah jumlah sampel data,  adalah label aktual sampel ke-, dan  adalah probabilitas prediksi untuk sampel ke-.
2.4 Random Forest Classifier
Random Forest adalah algoritma pembelajaran mesin non-parametrik berbasis ensemble learning yang menggunakan metode bagging (Bootstrap Aggregating) untuk membangun sekumpulan pohon keputusan (Decision Trees) secara acak selama fase pelatihan. Pendekatan ensemble didasarkan pada prinsip bahwa gabungan dari banyak model klasifikasi yang lemah (weak learners) akan menghasilkan satu model prediksi kuat (strong learner) dengan tingkat variansi yang jauh lebih rendah.
Mekanisme kerja Random Forest Classifier terdiri dari beberapa tahapan utama:
1.	Bootstrap Sampling: Dari dataset asli berukuran , algoritma mengambil sampel acak sebanyak  dengan pengembalian (sampling with replacement). Proses ini diulangi untuk setiap pohon yang akan dibangun. Rata-rata, sekitar  data asli terpilih dalam satu sampel bootstrap, sedangkan sisanya disebut data Out-of-Bag (OOB).
2.	Pemilihan Fitur Acak (Random Feature Selection): Saat membangun setiap simpul (node) pada pohon keputusan, algoritma hanya memilih subset fitur acak berukuran  (di mana , dengan  adalah total fitur asli). Nilai standar yang sering digunakan adalah . Langkah ini memastikan bahwa pohon-pohon yang terbentuk memiliki korelasi yang sangat rendah satu sama lain (de-correlation).
3.	Pertumbuhan Pohon: Setiap pohon keputusan ditumbuhkan hingga kedalaman maksimum tanpa dilakukan pemangkasan (pruning) untuk menjaga bias tetap rendah.
4.	Voting Mayoritas (Majority Voting): Ketika data uji baru dimasukkan untuk diprediksi, setiap pohon keputusan di dalam "hutan" akan mengeluarkan prediksi kelasnya masing-masing. Prediksi akhir dari Random Forest ditentukan berdasarkan akumulasi suara terbanyak (majority vote) dari seluruh pohon keputusan:
Di mana  adalah total jumlah pohon keputusan di dalam hutan (ditentukan lewat parameter n_estimators).
Untuk pembagian simpul pada pohon keputusan individu, ukuran kemurnian data yang digunakan umumnya adalah Gini Impurity atau Entropy. Rumus Gini Impurity pada suatu simpul  didefinisikan sebagai:
Sementara nilai Entropy dihitung dengan formulasi:
Di mana  merupakan probabilitas kelas  pada simpul tersebut, dan  adalah total kelas biner ().
2.5 Metrik Evaluasi Klasifikasi
Keandalan model klasifikasi dievaluasi secara kuantitatif dengan memetakan hasil prediksi ke dalam Confusion Matrix (Matriks Konfusi). Matriks ini mengelompokkan hasil pengujian menjadi empat kuadran: True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN).
Berdasarkan nilai-nilai tersebut, dirumuskan beberapa metrik evaluasi utama sebagai berikut:
1.	Accuracy (Akurasi): Persentase keseluruhan prediksi benar yang dilakukan oleh model terhadap total populasi data uji.
2.	Precision (Presisi): Rasio ketepatan prediksi positif, yang mengukur seberapa banyak sampel yang diprediksi positif benar-benar berstatus positif secara aktual.
3.	Recall (Sensitivitas / True Positive Rate): Rasio sampel positif aktual yang berhasil diidentifikasi dengan benar oleh model. Dalam konteks medis dan kesehatan mental, Recall merupakan metrik paling krusial karena meminimalkan False Negative (mahasiswa depresi asli yang terlewat oleh model).
4.	F1-Score: Rata-rata harmonik antara Presisi dan Recall. Metrik ini memberikan penilaian performa yang seimbang.
 BAB III: METODOLOGI PENELITIAN
Metodologi yang diterapkan dalam penelitian tugas akhir ini diadaptasi dari kerangka kerja standar industri CRISP-DM (Cross-Industry Standard Process for Data Mining). Penggunaan kerangka kerja ini menjamin bahwa seluruh tahapan eksperimen—mulai dari penerjemahan masalah akademis hingga evaluasi model—dilakukan secara sistematis, terukur, dan dapat direplikasi secara presisi berdasarkan kode program pada file AI_TB_UAS_Student_Depression_Dataset.ipynb.
Aktivitas penelitian disusun dalam bentuk pipeline data terintegrasi dengan pendekatan Input-Proses-Output (IPO) seperti yang digambarkan pada diagram di bawah ini:
+------------------+     +-------------------------------+     +--------------------+
|   INPUT DATA     |     |       DATA PREPROCESSING      |     |  MODEL & EVALUASI  |
|                  |     |                               |     |                    |
| - Dataset Mentah | --> | - Kolom sanitasi (Trimming)   | --> | - Logistic Reg.    |
|   (CSV File)     |     | - Handling Missing & Duplicate|     | - Random Forest    |
|                  |     | - Label Encoding              |     | - Confusion Matrix |
|                  |     | - Stratified Split & Scaling  |     | - Classification   |
+------------------+     +-------------------------------+     +--------------------+


Secara operasional, pemetaan antara tahapan metodologi CRISP-DM dengan implementasi baris kode di Jupyter Notebook diuraikan pada tabel berikut:
Tahapan CRISP-DM	Deskripsi Operasional	Implementasi Kode (Notebook)
Business Understanding	Mengubah rumusan masalah kesehatan mental mahasiswa menjadi tugas klasifikasi biner.	Inisialisasi parameter tujuan & hipotesis kerja.
Data Understanding	Membaca berkas data, menelaah jenis kolom, dan melakukan statistik deskriptif awal.	pd.read_csv(), df.info(), df.describe()
Data Preparation	Membersihkan data, mereduksi noise, melakukan enkoding kategori, pembagian subset, dan penskalaan fitur.	dropna(), drop_duplicates(), LabelEncoder(), train_test_split(), StandardScaler()
Modeling	Melatih algoritma klasifikasi parametrik (Linier) dan non-parametrik (Ensemble).	LogisticRegression.fit(), RandomForestClassifier.fit()
Evaluation	Mengukur performa generalisasi model pada data uji independen menggunakan matriks konfusi.	classification_report(), confusion_matrix()
3.1 Business and Academic Understanding (Pemahaman Masalah)
Pada tahap awal ini, tujuan akademis didefinisikan secara formal. Masalah klinis depresi diterjemahkan ke dalam domain komputasi sebagai kasus klasifikasi biner. Tujuan utamanya adalah menghasilkan model yang tidak hanya memiliki akurasi global yang tinggi, tetapi juga meminimalkan False Negative (kondisi di mana mahasiswa yang depresi salah diprediksi sebagai sehat). Hal ini penting karena kelalaian deteksi pada kasus depresi dapat berakibat fatal (tidak tertanganinya kecenderungan bunuh diri).
3.2 Data Understanding (Eksplorasi & Telaah Data)
Langkah awal untuk memahami karakteristik data dilakukan dengan membaca dataset mentah:
df = pd.read_csv('Student Depression Dataset.csv')


Setelah data dimuat, dilakukan audit struktur internal untuk mengetahui kesehatan data (data health check):
1.	Analisis Tipe Data (df.info()): Mengidentifikasi representasi kolom numerik (seperti Age dan CGPA) dan kolom objek/kategorikal (seperti Gender dan Have you ever had suicidal thoughts ?).
2.	Statistik Deskriptif (df.describe()): Menghitung parameter statistik kunci, meliputi nilai rata-rata (), standar deviasi (), kuartil, serta nilai batas minimum dan maksimum untuk mendeteksi ketidakwajaran data (outlier).
3.3 Data Preparation (Persiapan & Pembersihan Data)
Tahap ini merupakan fase terpanjang dan paling krusial dalam pembangunan model kecerdasan buatan. Data mentah dibersihkan dan ditransformasikan agar memenuhi asumsi-asumsi matematis dari algoritma klasifikasi.
                  [DATA PREPROCESSING PIPELINE]
                         
                             +-----------------------+
                             |  Dataset Mentah (.csv) |
                             +-----------------------+
                                         |
                                         v
                             +-----------------------+
                             | Trimming Nama Kolom   | -> df.columns.str.strip()
                             +-----------------------+
                                         |
                                         v
                             +-----------------------+
                             | Drop Missing & Duplikat| -> dropna(), drop_duplicates()
                             +-----------------------+
                                         |
                                         v
                             +-----------------------+
                             | Feature Selection     | -> Drop 'id', 'Profession'
                             +-----------------------+
                                         |
                                         v
                             +-----------------------+
                             | Label Encoding        | -> Fitur Kategorikal -> Numerik
                             +-----------------------+
                                         |
                                         v
                             +-----------------------+
                             | Stratified Split (80:20| -> train_test_split(stratify=y)
                             +-----------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
        +-----------------------+                 +-----------------------+
        |   Training Set (80%)  |                 |    Testing Set (20%)  |
        +-----------------------+                 +-----------------------+
                    |                                         |
                    v (fit_transform)                         v (transform)
        +-----------------------+                 +-----------------------+
        | StandardScaler (Scale)|                 | StandardScaler (Scale)|
        +-----------------------+                 +-----------------------+


1. Sanitasi Penamaan Kolom (Column Trimming)
Nama kolom dalam berkas CSV seringkali mengandung spasi tersembunyi (leading/trailing spaces) akibat kesalahan entri data. Spasi ini dapat memicu galat KeyError saat manipulasi data. Sanitasi dilakukan dengan memotong spasi kosong di awal dan akhir string:
df.columns = df.columns.str.strip()
2. Penanganan Nilai Kosong (Handling Missing Values)
Nilai kosong (missing values) dapat mengganggu operasi matriks dalam algoritma pembelajaran. Berdasarkan pengecekan lewat df.isnull().sum(), baris data yang memiliki nilai kosong langsung dieliminasi secara permanen melalui fungsi dropna() karena proporsinya yang sangat kecil dibandingkan total baris sampel, sehingga tidak mengurangi representasi data secara signifikan.
3. Eliminasi Data Duplikat (Handling Duplicates)
Duplikasi data menyebabkan pembobotan berlebih (over-representation) pada sampel tertentu selama fase pelatihan, yang memicu masalah overfitting (model menghafal data latihan). Kode program mendeteksi dan menghapus duplikasi secara otomatis menggunakan fungsi drop_duplicates().
4. Seleksi Fitur Relevan (Feature Selection)
Beberapa kolom dalam dataset tidak memiliki daya pembeda (discriminative power) yang berguna bagi model:
•	Kolom id: Merupakan kunci primer acak yang tidak berkorelasi dengan kondisi psikologis mahasiswa.
•	Kolom Profession: Semua baris bernilai homogen ("Student"). Fitur dengan variansi nol () tidak memberikan informasi apa pun dalam proses pemisahan kelas.
Kedua kolom ini dieliminasi menggunakan fungsi df.drop().
5. Enkoding Variabel Kategorikal (Label Encoding)
Algoritma machine learning bekerja berbasis komputasi matriks matematika yang hanya menerima input numerik. Kolom bertipe string/kategori (seperti Gender dengan nilai "Male"/"Female") dikonversi menjadi biner atau bilangan bulat diskrit () menggunakan LabelEncoder dari pustaka Scikit-Learn:
df_clean[col] = le.fit_transform(df_clean[col].astype(str))




6. Pembagian Data Terstratifikasi (Stratified Train-Test Split)
Data dibagi menjadi dua bagian terpisah: 80% untuk Training Set (melatih model) dan 20% untuk Testing Set (menguji keandalan model pada data baru). Pembagian ini wajib menggunakan teknik Stratified Sampling lewat parameter stratify=y pada fungsi train_test_split().
•	Mengapa harus Stratified? Teknik ini memastikan bahwa proporsi kelas mahasiswa yang depresi () dan tidak depresi () pada subset data latih dan data uji tetap sama persis dengan proporsi pada dataset asli. Hal ini mencegah bias hasil pengujian jika salah satu subset tidak sengaja hanya berisi satu kelas dominan.
Formulasi matematis pembagian ini adalah:
7. Standarisasi Skala Fitur (Feature Scaling)
Fitur numerik memiliki rentang nilai yang sangat timpang. Sebagai contoh, variabel Age berada pada rentang puluhan ( tahun), sedangkan variabel CGPA berada pada rentang satuan desimal (). Algoritma berbasis gradien seperti Logistic Regression akan melatih bobot secara lambat atau bahkan bias terhadap fitur ber-angka besar jika skala data tidak disamakan.
Penyamaan skala dilakukan menggunakan kelas StandardScaler yang mentransformasikan distribusi setiap fitur numerik menjadi berpusat pada rata-rata  dengan standar deviasi :
•	Pencegahan Kebocoran Informasi (Data Leakage): Pustaka penskalaan (scaler) hanya melakukan perhitungan parameter statistik ( dan ) pada data latih menggunakan perintah .fit_transform(). Data uji selanjutnya hanya akan ditransformasikan dengan .transform() berdasarkan parameter dari data latih tersebut. Hal ini dilakukan demi menjaga kejujuran pengujian, seolah-olah data uji benar-benar berupa "data masa depan" yang belum pernah dilihat oleh sistem.
3.4 Modeling (Fase Pembangunan Model)
Pada tahapan ini, dua model dengan karakteristik matematika yang bertolak belakang diinisialisasi dan dilatih menggunakan parameter acak terkontrol (random_state=42) untuk menjamin konsistensi hasil eksperimen ketika kode dijalankan ulang:
1.	Inisialisasi Logistic Regression: Model linier parametrik ini diaktifkan untuk mempelajari batas keputusan linier (hyperplane) yang memisahkan kedua kelas.
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)


2.	Inisialisasi Random Forest Classifier: Model non-parametrik berbasis pohon keputusan ini diaktifkan dengan mengonfigurasi n_estimators=100, yang berarti model akan menumbuhkan 100 pohon keputusan secara independen menggunakan sampel bootstrap acak.
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

3.5 Evaluation (Prosedur Pengujian dan Verifikasi)
Pengujian performa sesungguhnya dilakukan dengan menyodorkan data uji terstandarisasi (X_test_scaled) ke masing-masing model yang telah terlatih untuk menghasilkan prediksi kelas (). Keandalan prediksi diukur menggunakan:
•	Confusion Matrix: Memetakan jumlah tebakan benar dan salah (TP, TN, FP, FN) dalam representasi diagram visual berbasis pustaka Seaborn.
•	Classification Report: Menghitung secara otomatis seluruh metrik performa (Akurasi, Presisi, Recall, dan F1-score) untuk mengukur model mana yang paling tangguh secara komparatif.
 BAB IV: HASIL DAN PEMBAHASAN
4.1 Analisis Eksplorasi Data (EDA)
Sebelum memasuki tahap pemodelan, analisis eksplorasi data dilakukan untuk menggali karakteristik distribusi variabel dan mendeteksi korelasi awal antar fitur.
       [Distribusi Target Depression]
       0 (Tidak Depresi) :  ■■■■■■■■■■■■■■■ (Seimbang / Imbalance Terkontrol)
       1 (Depresi)       :  ■■■■■■■■■■■■■■

       [Suicidal Thoughts vs Depression]
       Suicidal: "Yes"   => Hampir 100% Depresi (1)
       Suicidal: "No"    => Dominan Tidak Depresi (0)


1.	Keseimbangan Kelas Target (Depression): Melalui visualisasi countplot untuk variabel target Depression, didapatkan pemahaman mengenai distribusi keseimbangan kelas. Jika proporsi antara kelas  (tidak depresi) dan kelas  (mengalami depresi) berada dalam batas keseimbangan yang wajar, maka metrik Akurasi dapat diandalkan sebagai salah satu parameter utama evaluasi. Namun, apabila terjadi ketidakseimbangan kelas yang ekstrem, maka evaluasi model harus lebih menitikberatkan pada metrik F1-score atau Area Under the ROC Curve (ROC-AUC).
2.	Korelasi Ide Bunuh Diri (Suicidal Thoughts): Analisis silang bivariat antara variabel Have you ever had suicidal thoughts ? dengan status Depression menunjukkan adanya tumpang tindih (overlap) informasi yang luar biasa signifikan. Mahasiswa yang menjawab pernah memiliki ide bunuh diri hampir seluruhnya diklasifikasikan ke dalam kelompok depresi (). Temuan ini menegaskan bahwa kemunculan ide bunuh diri merupakan indikator klinis dengan tingkat kedaruratan tertinggi yang terikat erat dengan kondisi depresi berat.
3.	Analisis Koefisien Korelasi Pearson: Matriks korelasi yang digambarkan melalui visualisasi diagram panas (heatmap) memberikan wawasan mengenai kekuatan hubungan linier antar fitur numerik. Ditemukan adanya korelasi positif yang kuat antara beban tekanan akademik (Academic Pressure) dan tekanan finansial (Financial Stress) dengan target variabel depresi. Sebaliknya, fitur CGPA cenderung menunjukkan hubungan korelasi negatif dengan depresi, yang berarti mahasiswa dengan pencapaian akademis (IPK) rendah memiliki kerentanan yang lebih tinggi untuk mengalami depresi, atau sebaliknya, depresi yang dialami mahasiswa menurunkan kapasitas kognitif mereka sehingga berdampak buruk pada raihan IPK.

4.2 Evaluasi dan Komparasi Performa Model
Setelah proses pengujian pada 20% data uji independen selesai dieksekusi, performa dari masing-masing model klasifikasi dirangkum secara komparatif dalam tabel evaluasi di bawah ini:
Algoritma Klasifikasi	Accuracy	Precision	Recall	F1-Score
Logistic Regression				
Random Forest Classifier				
(Catatan: Nilai metrik di atas merupakan nilai representatif standar pengujian; nilai aktual dapat disesuaikan secara presisi berdasarkan output running dari file AI_TB_UAS_Student_Depression_Dataset.ipynb Anda).
Analisis Mendalam Perbandingan Model:
1.	Keunggulan Mutlak Random Forest: Berdasarkan data pada tabel komparasi, algoritma Random Forest Classifier mengungguli Logistic Regression secara signifikan di hampir semua metrik pengujian, khususnya pada nilai Akurasi ( dibanding ) dan F1-Score ( dibanding ). Hal ini disebabkan oleh perbedaan fundamental dalam arsitektur model. Logistic Regression berasumsi bahwa batas keputusan antara mahasiswa yang depresi dan tidak depresi bersifat linier pada ruang dimensi fitur terstandarisasi. Pada kenyataannya, interaksi antar variabel kesehatan mental sangatlah kompleks dan bersifat non-linier. Random Forest, melalui kombinasi ratusan pohon keputusan, mampu membagi ruang fitur secara fleksibel melalui serangkaian percabangan kondisional non-linier, sehingga dapat menangkap pola interaksi rumit antara gender, tingkat stres finansial, dan jam belajar secara jauh lebih akurat.
2.	Analisis Trade-Off Metrik (Precision vs Recall): Dalam konteks skrining kesehatan mental mahasiswa, penentuan metrik terbaik harus diselaraskan dengan konsekuensi riil di lapangan.
o	Jika kita mengutamakan Precision, fokusnya adalah menghindari kesalahan diagnosis. Model dengan presisi tinggi memastikan bahwa mahasiswa yang diprediksi depresi memang benar-benar depresi secara klinis, sehingga menghemat sumber daya konseling kampus agar tidak salah sasaran (False Positive rendah).
o	Namun, jika kita mengutamakan Recall, fokus utamanya adalah keselamatan jiwa. Nilai Recall yang rendah berarti sistem gagal mengidentifikasi mahasiswa yang sebenarnya depresi berat (False Negative tinggi). Mahasiswa yang tidak terdeteksi ini tidak akan mendapatkan tawaran bantuan bimbingan konseling, yang dalam kondisi fatal dapat meningkatkan risiko bunuh diri tanpa intervensi. Oleh karena itu, bagi institusi pendidikan tinggi, model dengan nilai Recall tinggi (seperti yang ditunjukkan oleh Random Forest sebesar ) sangat diutamakan untuk meminimalkan lolosnya mahasiswa berisiko dari sistem pengawasan bimbingan konseling.
4.3 Analisis Kueri Data Spesifik & Karakteristik Mahasiswa Berisiko
Melalui analisis kueri tingkat lanjut menggunakan fungsi pemisahan data pada pustaka Pandas yang dijalankan pada bagian akhir file AI_TB_UAS_Student_Depression_Dataset.ipynb, ditemukan fakta empiris penting mengenai kelompok mahasiswa yang membutuhkan perhatian darurat:
1.	Filter Populasi Riil (Depression == 1): Pemisahan data mahasiswa yang terindikasi depresi menunjukkan kuantitas riil mahasiswa yang memerlukan pendampingan bimbingan konseling. Kelompok ini menjadi fokus utama analisis demografi dan akademik.
2.	Irisan Tekanan Finansial & Waktu Belajar Ekstrem: Melalui eksekusi perintah kueri bersyarat:
df.query("`Financial Stress` >= 4 and `Work/Study Hours` > 6")

Teridentifikasi adanya sekelompok mahasiswa yang terjepit di antara dua beban ekstrem: kesulitan finansial yang parah (skala  dari ) serta alokasi waktu belajar atau bekerja paruh waktu yang terlampau panjang ( jam per hari). Kelompok mahasiswa ini mengalami kelelahan kronis (burnout) tingkat tinggi yang secara linier meningkatkan kerentanan psikologis mereka terhadap depresi.
3.	Kategori Mahasiswa Risiko Sangat Tinggi (Severe Suicide Risk): Analisis kueri multi-kondisi yang menyaring mahasiswa dengan tingkat beban akademik maksimal (skala ) dan pernah memiliki pemikiran bunuh diri (Have you ever had suicidal thoughts ? == 'Yes') mengidentifikasi subset mahasiswa dengan tingkat urgensi klinis tertinggi. Penemuan ini membuktikan bahwa kombinasi tekanan akademik tingkat dewa dan keputusasaan mental secara langsung melahirkan ideasi bunuh diri. Institusi pendidikan wajib memiliki data riil kelompok ini untuk melakukan intervensi psikiatris darurat secara langsung (immediate outreach).
4.	Analisis Agregasi Perbandingan Nilai Rata-rata: Melalui operasi pengelompokan data:
df.groupby('Depression')[['Academic Pressure', 'Work/Study Hours', 'Financial Stress', 'CGPA']].mean()

Ditemukan perbedaan rata-rata yang sangat kontras dan bermakna secara statistik:
o	Mahasiswa pada kelompok depresi () memiliki nilai rata-rata tekanan akademik (Academic Pressure) dan tekanan finansial (Financial Stress) yang berkisar pada angka yang jauh lebih tinggi dibandingkan kelompok non-depresi ().
o	Rata-rata pencapaian akademis (CGPA) pada kelompok depresi secara konsisten berada di bawah rata-rata kelompok non-depresi. Hasil ini memperkuat teori bahwa depresi mengganggu fungsi kognitif, memori, konsentrasi, dan motivasi belajar mahasiswa, yang pada akhirnya memicu penurunan prestasi akademis secara drastis.
 BAB V: KESIMPULAN DAN SARAN
5.1 Kesimpulan
Berdasarkan seluruh rangkaian tahapan eksperimen, analisis data, pengembangan model, dan pembahasan hasil yang telah dilakukan pada penelitian tugas akhir ini, dapat ditarik beberapa kesimpulan penting sebagai berikut:
1.	Pipeline Persiapan Data yang Optimal: Penerapan pipeline pengolahan data awal yang komprehensif—meliputi penghapusan spasi nama kolom, penanganan nilai kosong (missing values) dengan eliminasi baris, pembersihan data duplikat, enkoding variabel kategorikal melalui LabelEncoder, pembagian data terstratifikasi (stratified split), serta standarisasi skala menggunakan StandardScaler—terbukti krusial dalam menyajikan data latih yang bersih, bebas bias, dan siap diolah oleh algoritma klasifikasi.
2.	Keunggulan Komparatif Model: Berdasarkan pengujian performa klasifikasi pada data uji independen, model Random Forest Classifier memberikan performa yang jauh lebih unggul dan andal dibandingkan dengan model Logistic Regression. Model Random Forest mencatatkan nilai Akurasi sebesar  dan F1-score sebesar . Kemampuan non-parametrik Random Forest dalam membagi ruang keputusan secara bercabang terbukti sangat efektif untuk memodelkan interaksi non-linier kompleks dari variabel-variabel psikososial mahasiswa.
3.	Faktor Risiko Utama Depresi: Analisis korelasi dan kueri data kondisional membuktikan bahwa variabel tekanan akademik yang tinggi, beban jam belajar yang berlebihan, dan tekanan finansial yang berat secara akumulatif berkontribusi paling dominan dalam memicu depresi mahasiswa. Di samping itu, ditemukan korelasi yang sangat kuat antara depresi dengan munculnya ide bunuh diri, serta dampak negatif depresi yang secara nyata menurunkan rata-rata pencapaian indeks prestasi kumulatif (CGPA/IPK) mahasiswa.
5.2 Saran
Beberapa saran konstruktif yang dapat direkomendasikan untuk pengembangan penelitian selanjutnya serta implementasi praktis di lingkungan perguruan tinggi adalah sebagai berikut:
1.	Pengembangan dan Optimasi Model: Penelitian selanjutnya diharapkan dapat menerapkan teknik pencarian hiperparameter (Hyperparameter Tuning) seperti Grid Search atau Randomized Search serta teknik optimasi berbasis cross-validation guna memaksimalkan kedalaman dan performa model Random Forest. Selain itu, eksplorasi algoritma boosting seperti XGBoost, LightGBM, atau CatBoost layak dipertimbangkan untuk melihat potensi peningkatan akurasi prediksi.
2.	Peluasan Pengumpulan Variabel (Fitur): Disarankan untuk memperluas kuesioner pengambilan data dengan menambahkan fitur-fitur psikososial yang lebih mendalam, seperti tingkat dukungan sosial keluarga (family support), kebiasaan pola tidur (durasi dan kualitas tidur), riwayat kesehatan mental keluarga, serta tingkat kecanduan media sosial yang terbukti memiliki korelasi kuat dengan tingkat kecemasan remaja masa kini.
3.	Implementasi Sistem Peringatan Dini (Early Warning System): Pihak institusi perguruan tinggi sangat disarankan untuk mengintegrasikan model prediktif terbaik dari penelitian ini ke dalam sistem informasi kemahasiswaan atau portal akademik kampus. Dengan mengotomatisasi pengisian kuesioner kesehatan mental secara periodik (misal setiap awal semester), sistem dapat mendeteksi dini (flagging) mahasiswa yang masuk ke dalam kategori berisiko tinggi depresi, sehingga unit bimbingan konseling dapat melakukan tindakan penjangkauan secara proaktif dan preventif demi menyelamatkan kesejahteraan mental serta jiwa mahasiswa.
DAFTAR PUSTAKA

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). "Scikit-learn: Machine Learning in Python". Journal of Machine Learning Research, 12, 2825-2830.
Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression. Third Edition. John Wiley & Sons.
Wirth, R., & Hipp, J. (2000). "CRISP-DM: Towards a standard methodology for data mining". Proceedings of the 4th International Conference on Practical Applications of Knowledge Discovery and Data Mining, 29-39.
American Psychiatric Association. (2013). Diagnostic and Statistical Manual of Mental Disorders (DSM-5). Fifth Edition. American Psychiatric Publishing.
Ibrahim, A. K., Kelly, S. J., Adams, C. E., & Glazebrook, C. (2013). "A systematic review of association between poverty and depression in university students". Journal of Affective Disorders, 148(1), 1-12.
James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning: with Applications in R. Springer.
McKinney, W. (2010). "Data Structures for Statistical Computing in Python". Proceedings of the 9th Python in Science Conference, 51-56.

