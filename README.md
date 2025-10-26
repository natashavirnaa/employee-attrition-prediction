# Laporan Proyek Machine Learning 

## Employee Attrition Prediction

**Natasha Virna Arthilita** — *5003231020*  
**Violita Inggar Ariana** — *5003231021*  
**Fahmadina Sophia** — *5003231122*

## 🗂️ Daftar Isi

1. [Domain Proyek: Manajemen Sumber Daya](#1-domain-proyek-manajamen-sumber-daya)  
   - [Referensi](#referensi)  
2. [Business Understanding](#2-business-understanding)  
   - [Problem Statements](#problem-statements)  
   - [Goals](#goals)  
   - [Solution Statements](#solution-statements)  
   - [Project Benefits](#project-benefits)  
3. [Data Understanding](#3-data-understanding)  
   - [Sumber Data](#sumber-data)  
   - [Deskripsi Fitur](#deskripsi-fitur)  
   - [Penjelasan Kontekstual Fitur](#penjelasan-kontekstual-fitur)  
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
     - [Missing Value & Outliers](#missing-value--outliers)  
     - [Univariate Analysis](#univariate-analysis)  
     - [Multivariate Analysis](#multivariate-analysis)  
     - [Kesimpulan EDA](#kesimpulan-eda)  
4. [Data Preparation](#4-data-preparation)  
   - [Label Encoding dengan Mapping pada Fitur Target](#label-encoding-dengan-mapping-pada-fitur-target)  
   - [Splitting Dataset](#splitting-dataset)  
   - [Feature Engineering, Data Cleaning and Preprocessing](#feature-engineering-data-cleaning-and-preprocessing)  
5. [Model Training, Comparison, Selection and Tuning](#5-model-training-comparison-selection-and-tuning)  
   - [Model Selection](#model-selection)  
   - [Feature Selection](#feature-selection)  
   - [Hyperparameter Tuning](#hyperparameter-tuning)  
6. [Model Testing and Evaluation](#6-model-testing-and-evaluation)  
   - [Data Test Predict](#data-test-predict)  
   - [Best Model Evaluation](#best-model-evaluation)  
     - [Classification Report](#classification-report)  
     - [Metode Evaluasi Lanjutan](#metode-evaluasi-lanjutan)  
     - [Confusion Matrix](#confusion-matrix)  
     - [Plot ROC-AUC Curve](#plot-roc-auc-curve)  
     - [Plot PR-AUC Curve](#plot-pr-auc-curve)  
7. [Save Best Model](#7-save-best-model)  
8. [Model Interpretation](#8-model-interpretation)  
   - [Interpretation with SHAP Values](#interpretation-with-shap-values)  
   - [Feature Importance](#feature-importance)  
9. [Financial Result](#9-financial-result)  
10. [Conclusions](#10-conclusions)  
    - [Ringkasan Proyek](#ringkasan-proyek)  
    - [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)  
    - [Penanganan Ketidakseimbangan Data](#penanganan-ketidakseimbangan-data)  
    - [Interpretasi dan Validasi Model](#interpretasi-dan-validasi-model)  
    - [Estimasi Nilai Finansial](#estimasi-nilai-finansial)  
    - [Langkah Selanjutnya](#langkah-selanjutnya)

## Domain Proyek : Manajemen Sumber Daya
![Employee Attrition](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/employe-attrition.png?raw=true)
Di dunia bisnis modern, pengelolaan sumber daya manusia menjadi faktor kunci bagi keberlanjutan dan daya saing perusahaan [1]. Salah satu tantangan utama dalam bidang ini adalah tingginya tingkat attrition atau employee turnover, yaitu kondisi ketika karyawan meninggalkan perusahaan baik secara sukarela maupun tidak. Tingkat turnover yang tinggi dapat menimbulkan biaya besar bagi perusahaan, mengganggu produktivitas, serta berdampak negatif terhadap moral tim [2].
Untuk mengatasinya, banyak organisasi kini mengandalkan pendekatan berbasis data guna memahami faktor-faktor yang memengaruhi keputusan karyawan dalam bertahan atau keluar dari perusahaan [3]. Dengan memanfaatkan analisis prediktif, perusahaan dapat mengidentifikasi pola perilaku karyawan, mendeteksi risiko turnover lebih dini, serta merancang strategi retensi yang lebih efektif.
Dataset yang digunakan dalam proyek ini mencakup berbagai aspek profil karyawan, lingkungan kerja, dan status karyawan (bertahan atau keluar). Variabel-variabel tersebut meliputi faktor demografis, jabatan, jam lembur, tingkat kepuasan kerja, dan beban kerja, yang dapat digunakan untuk membangun model prediktif employee attrition dan mengeksplorasi fitur-fitur yang paling berpengaruh terhadap keputusan karyawan [4].

---

**Referensi**
<div style="font-size: 10px; line-height: 1.6; margin-left: 10px;">
[1] Dessler, G. (2020). <i>Human Resource Management</i> (16th ed.). Pearson Education.<br>
[2] Cascio, W. F. (2015). <i>Managing Human Resources: Productivity, Quality of Work Life, Profits</i> (10th ed.). McGraw-Hill Education.<br>
[3] Nguyen, T., & Duong, M. (2021). “Predicting Employee Turnover Using Machine Learning Techniques.” <i>Journal of Human Resource Analytics,</i> 5(3), 45–57.<br>
[4] IBM HR Analytics Employee Attrition & Performance Dataset (2019). <i>Kaggle.</i><br>
<a href="https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset">https://www.kaggle.com/datasets/pavansubhashh/ibm-hr-analytics-attrition-dataset</a>
</div>

---

## Business Understanding
### Problem Statemnents
Dalam dunia kerja modern, setiap perusahaan berusaha mempertahankan karyawan berprestasi untuk menjaga stabilitas operasional dan daya saing. Namun, tingginya tingkat employee attrition atau turnover menjadi tantangan serius bagi manajemen sumber daya manusia. Fenomena ini dapat menyebabkan peningkatan biaya rekrutmen dan pelatihan, penurunan produktivitas, serta menurunnya moral kerja tim secara keseluruhan.

Berdasarkan hal tersebut, pernyataan masalah yang diangkat dalam proyek ini adalah sebagai berikut: 
1. **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor utama yang memengaruhi keputusan karyawan untuk keluar dari perusahaan?  
2. **Pernyataan Masalah 2:** Bagaimana membangun model prediktif yang mampu memperkirakan kemungkinan seorang karyawan akan keluar dengan akurasi yang tinggi?  
3. **Pernyataan Masalah 3:** Bagaimana memanfaatkan hasil analisis tersebut untuk merumuskan strategi retensi karyawan yang efektif dan berbasis data?

### Goals
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini adalah:  
1. **Tujuan 1:** Melakukan eksplorasi dan analisis terhadap data karyawan untuk menemukan pola dan variabel yang berkorelasi dengan perilaku attrition.
2. **Tujuan 2:** Membangun model prediktif berbasis machine learning untuk menghitung probabilitas seorang karyawan akan keluar dari perusahaan.
3. **Tujuan 3:** Menyediakan insight dan rekomendasi strategis yang dapat membantu manajemen dalam meningkatkan retensi karyawan dan menciptakan lingkungan kerja yang lebih produktif.

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang akan diterapkan meliputi:  
- **Eksperimen dengan Berbagai Algoritma Klasifikasi**
  Membangun dan membandingkan performa beberapa algoritma seperti:
  - Logistic Regression
  - Random Forest
  - XGBoost / LightGBM
- **Optimasi Model dengan Hyperparameter Tuning**
  Menggunakan pendekatan seperti Grid Search atau Optuna (Bayesian Optimization) untuk menemukan konfigurasi parameter terbaik.
- **Evaluasi Model dengan Metrik yang Relevan**
  Menerapkan metrik evaluasi seperti:
  - Accuracy — untuk mengukur kinerja keseluruhan model.
  - Precision, Recall, dan F1-Score — untuk menilai performa pada kelas minoritas (attrition).
  - ROC-AUC — untuk mengevaluasi kemampuan model dalam membedakan karyawan yang bertahan dan keluar.
  - Confusion Matrix — untuk memvisualisasikan hasil prediksi model.
- **Analisis Fitur Dan visualisasi**
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi pada analisis.

### Project Benefit
Dengan implementasi solusi ini, manfaat yang diharapkan antara lain:
- **Efisiensi Biaya SDM:** Mengurangi biaya rekrutmen dan pelatihan akibat tingginya turnover.
- **Peningkatan Retensi Karyawan:** Mengidentifikasi dan menindaklanjuti karyawan dengan risiko tinggi untuk keluar.
- **Pengambilan Keputusan Berbasis Data:** Membantu manajemen dalam menyusun strategi retensi dan kebijakan HR yang lebih tepat sasaran.
- **Perbaikan Lingkungan Kerja:** Memberikan insight terhadap faktor-faktor yang menyebabkan ketidakpuasan dan menurunkan motivasi kerja.
- **Produktivitas yang Lebih Stabil:** Menjaga kontinuitas tim dan mengurangi dampak negatif dari pergantian karyawan.

---
## Data Understanding
### Sumber Data
Dataset yang digunakan dalam proyek ini diperoleh dari situs Kaggle. Dataset ini berisi data profil dan lingkungan kerja karyawan yang digunakan untuk memprediksi kemungkinan attrition atau keluar dari perusahaan.

Dataset terdiri dari dua berkas utama, yaitu train.csv dengan 1.173 data latih dan test.csv dengan 295 data uji. Setiap data memuat informasi mengenai berbagai variabel yang berpotensi memengaruhi keputusan karyawan untuk tetap bertahan atau keluar.

Secara keseluruhan, terdapat **35 variabel**, termasuk satu variabel target yaitu Attrition. Variabel ini menunjukkan apakah seorang karyawan keluar (1) atau tetap bekerja (0). Sebagian kecil karyawan termasuk dalam kategori attrition, sehingga terdapat ketidakseimbangan kelas *(class imbalance)* yang menjadi tantangan dalam proses pelatihan model prediktif.  

## Deskripsi Fitur
| Nama Fitur | Deskripsi | Tipe Data |
|-------------|------------|-----------|
| id | ID unik karyawan untuk identifikasi. | `object` |
| Age | Usia karyawan. | `int64` |
| BusinessTravel | Frekuensi perjalanan dinas karyawan. | `object` |
| DailyRate | Gaji harian. | `int64` |
| Department | Departemen tempat karyawan bekerja. | `object` |
| DistanceFromHome | Jarak tempat tinggal karyawan ke kantor. | `int64` |
| Education | Tingkat pendidikan (1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctor). | `int64` |
| EducationField | Bidang studi terakhir karyawan. | `object` |
| EmployeeCount | Jumlah karyawan. | `int64` |
| EmployeeNumber | Nomor unik karyawan dalam sistem HR. | `int64` |
| EnvironmentSatisfaction | Tingkat kepuasan terhadap lingkungan kerja (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| Gender | Jenis kelamin karyawan. | `object` |
| HourlyRate | Upah per jam. | `int64` |
| JobInvolvement | Tingkat keterlibatan pekerjaan (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| JobLevel | Level jabatan karyawan. | `int64` |
| JobRole | Posisi/jabatan spesifik karyawan. | `object` |
| JobSatisfaction | Tingkat kepuasan pekerjaan (1 = Low, 2 = Medium, 3 = High, 4 = Very High). | `int64` |
| MaritalStatus | Status pernikahan karyawan. | `object` |
| MonthlyIncome | Gaji bulanan karyawan. | `int64` |
| MonthlyRate | Tarif bulanan karyawan. | `int64` |
| NumCompaniesWorked | Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya. | `int64` |
| Over18 | Status usia di atas 18 tahun. | `object` |
| OverTime | Status lembur karyawan. | `object` |
| PercentSalaryHike | Persentase kenaikan gaji tahunan terakhir. | `int64` |
| PerformanceRating | Penilaian kinerja terakhir. | `int64` |
| RelationshipSatisfaction | Tingkat kepuasan terhadap hubungan kerja (1 = Low, 2 = Good, 3 = Excellent, 4 = Outstanding). | `int64` |
| StandardHours | Jam kerja standar (selalu 80 dalam dataset). | `int64` |
| StockOptionLevel | Level kepemilikan saham perusahaan. | `int64` |
| TotalWorkingYears | Total tahun pengalaman kerja. | `int64` |
| TrainingTimesLastYear | Jumlah pelatihan yang diikuti selama setahun terakhir. | `int64` |
| WorkLifeBalance | Tingkat keseimbangan kerja-hidup (1 = Bad, 2 = Good, 3 = Better, 4 = Best). | `int64` |
| YearsAtCompany | Total tahun bekerja di perusahaan saat ini. | `int64` |
| YearsInCurrentRole | Total tahun di posisi/jabatan saat ini. | `int64` |
| YearsSinceLastPromotion | Tahun sejak promosi terakhir. | `int64` |
| YearsWithCurrManager | Tahun bekerja dengan manajer saat ini. | `int64` |
| Attrition | Target: apakah karyawan keluar dari perusahaan (1 = Yes, 0 = No). | `object` |

### [Exploratory Data Analysis] - Statistika Deskriptif
| Fitur | count | mean | std | min | 25% | 50% | 75% | max |
|-------|--------|------|------|------|------|------|------|------|
| Age | 1176.0 | 37.00 | 9.18 | 18.0 | 30.0 | 36.0 | 43.0 | 60.0 |
| DailyRate | 1176.0 | 803.99 | 401.34 | 103.0 | 467.75 | 799.5 | 1157.0 | 1499.0 |
| DistanceFromHome | 1176.0 | 9.37 | 8.18 | 1.0 | 2.0 | 7.0 | 14.0 | 29.0 |
| Education | 1176.0 | 2.91 | 1.03 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
| EmployeeCount | 1176.0 | 1.00 | 0.00 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| EmployeeNumber | 1176.0 | 1015.83 | 599.66 | 1.0 | 487.75 | 1004.5 | 1547.25 | 2062.0 |
| EnvironmentSatisfaction | 1176.0 | 2.72 | 1.09 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| HourlyRate | 1176.0 | 65.50 | 20.37 | 30.0 | 48.0 | 66.0 | 83.0 | 100.0 |
| JobInvolvement | 1176.0 | 2.74 | 0.70 | 1.0 | 2.0 | 3.0 | 3.0 | 4.0 |
| JobLevel | 1176.0 | 2.08 | 1.09 | 1.0 | 1.0 | 2.0 | 3.0 | 5.0 |
| JobSatisfaction | 1176.0 | 2.72 | 1.11 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| MonthlyIncome | 1176.0 | 6544.02 | 4653.74 | 1009.0 | 2948.0 | 5004.5 | 8420.5 | 19973.0 |
| MonthlyRate | 1176.0 | 14390.24 | 7192.83 | 2094.0 | 8051.0 | 14373.0 | 20770.75 | 26999.0 |
| NumCompaniesWorked | 1176.0 | 2.69 | 2.49 | 0.0 | 1.0 | 2.0 | 4.0 | 9.0 |
| PercentSalaryHike | 1176.0 | 15.24 | 3.68 | 11.0 | 12.0 | 14.0 | 18.0 | 25.0 |
| PerformanceRating | 1176.0 | 3.16 | 0.36 | 3.0 | 3.0 | 3.0 | 3.0 | 4.0 |
| RelationshipSatisfaction | 1176.0 | 2.74 | 1.09 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 |
| StandardHours | 1176.0 | 80.00 | 0.00 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 |
| StockOptionLevel | 1176.0 | 0.79 | 0.85 | 0.0 | 0.0 | 1.0 | 1.0 | 3.0 |
| TotalWorkingYears | 1176.0 | 11.36 | 7.80 | 0.0 | 6.0 | 10.0 | 15.0 | 40.0 |
| TrainingTimesLastYear | 1176.0 | 2.76 | 1.26 | 0.0 | 2.0 | 3.0 | 3.0 | 6.0 |
| WorkLifeBalance | 1176.0 | 2.76 | 0.72 | 1.0 | 2.0 | 3.0 | 3.0 | 4.0 |
| YearsAtCompany | 1176.0 | 7.05 | 6.09 | 0.0 | 3.0 | 5.0 | 10.0 | 37.0 |
| YearsInCurrentRole | 1176.0 | 4.23 | 3.57 | 0.0 | 2.0 | 3.0 | 7.0 | 17.0 |
| YearsSinceLastPromotion | 1176.0 | 2.18 | 3.22 | 0.0 | 0.0 | 1.0 | 3.0 | 15.0 |
| YearsWithCurrManager | 1176.0 | 4.20 | 3.56 | 0.0 | 2.0 | 3.0 | 7.0 | 17.0 |
| Attrition | 1176.0 | 0.16 | 0.37 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

Dataset ini mencakup **1.176 karyawan** dengan rata-rata usia **37 tahun**, yang menunjukkan dominasi **usia produktif** di lingkungan kerja. Pendapatan bulanan rata-rata mencapai **Rp6.544**, dengan variasi yang cukup besar antar karyawan. Rata-rata masa kerja karyawan di perusahaan ini sekitar **7 tahun**, sementara total pengalaman kerja mereka mencapai **11 tahun**.

Tingkat **kepuasan terhadap pekerjaan**, **lingkungan kerja**, dan **hubungan antar rekan** berada di kisaran **2,7 dari 4**, yang mengindikasikan tingkat kepuasan sedang. Sebagian besar karyawan pernah bekerja di sekitar **tiga perusahaan sebelumnya**, dengan rata-rata **kenaikan gaji tahunan sebesar 15%**.

Proporsi karyawan yang keluar dari perusahaan (**Attrition = 1**) tercatat sebesar **16%**, menunjukkan tingkat **turnover yang relatif rendah**, namun tetap **penting untuk dianalisis lebih lanjut**. Secara keseluruhan, data ini menggambarkan **profil tenaga kerja yang relatif stabil**, tetapi dengan variasi dalam **pendapatan**, **masa kerja**, dan **kepuasan kerja** yang berpotensi memengaruhi keputusan karyawan untuk **bertahan atau meninggalkan perusahaan**.

### Perbandingan Rata-Rata Variabel Berdasarkan Status Attrition  
| Fitur | Attrition = 0 | Attrition = 1 |
|-------|----------------|----------------|
| Age | 37.74 | 33.13 |
| DailyRate | 816.56 | 738.74 |
| DistanceFromHome | 9.05 | 10.97 |
| Education | 2.92 | 2.84 |
| EmployeeCount | 1.00 | 1.00 |
| EmployeeNumber | 1009.80 | 1047.15 |
| EnvironmentSatisfaction | 2.77 | 2.44 |
| HourlyRate | 65.67 | 64.60 |
| JobInvolvement | 2.78 | 2.54 |
| JobLevel | 2.16 | 1.66 |
| JobSatisfaction | 2.77 | 2.46 |
| MonthlyIncome | 6871.64 | 4843.88 |
| MonthlyRate | 14321.50 | 14747.10 |
| NumCompaniesWorked | 2.64 | 2.97 |
| PercentSalaryHike | 15.25 | 15.16 |
| PerformanceRating | 3.16 | 3.16 |
| RelationshipSatisfaction | 2.76 | 2.63 |
| StandardHours | 80.00 | 80.00 |
| StockOptionLevel | 0.84 | 0.52 |
| TotalWorkingYears | 12.00 | 8.06 |
| TrainingTimesLastYear | 2.78 | 2.65 |
| WorkLifeBalance | 2.78 | 2.63 |
| YearsAtCompany | 7.46 | 4.90 |
| YearsInCurrentRole | 4.49 | 2.87 |
| YearsSinceLastPromotion | 2.24 | 1.87 |
| YearsWithCurrManager | 4.46 | 2.83 |

Karyawan yang **keluar dari perusahaan (Attrition = 1)** umumnya memiliki **usia lebih muda**, dengan rata-rata sekitar **33 tahun**, dibandingkan karyawan yang bertahan (**37 tahun**). Mereka juga memiliki **pendapatan bulanan** dan **tingkat jabatan** yang lebih rendah, serta **masa kerja yang lebih singkat** di perusahaan. Selain itu, **tingkat keterlibatan kerja**, **kepuasan terhadap pekerjaan**, dan **kepuasan terhadap lingkungan kerja** cenderung lebih rendah pada kelompok ini.

Karyawan yang keluar juga memiliki **total pengalaman kerja** dan **masa bekerja bersama manajer** yang lebih pendek, serta **kesempatan promosi** yang lebih sedikit. Meskipun perbedaan dalam **kenaikan gaji tahunan** dan **work-life balance** relatif kecil, pola ini menunjukkan bahwa **faktor kepuasan kerja** dan **peluang pengembangan karier** menjadi pendorong utama dalam keputusan karyawan untuk **meninggalkan perusahaan**.

### [Exploratory Data Analysis] - Menangani Missing Value dan Outliers
Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap duplikasi data dan *missing value*. Hasilnya menunjukkan bahwa tidak terdapat duplikasi data maupun *missing value* di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.
| Nama Fitur | Data Train |
|-------------|------------|
| id | 0 |
| Age | 0 |
| BusinessTravel | 0 |
| DailyRate | 0 |
| Department | 0 |
| DistanceFromHome | 0 |
| Education | 0 |
| EducationField | 0 |
| EmployeeCount | 0 |
| EmployeeNumber | 0 |
| EnvironmentSatisfaction | 0 |
| Gender | 0 |
| HourlyRate | 0 |
| JobInvolvement | 0 |
| JobLevel | 0 |
| JobRole | 0 |
| JobSatisfaction | 0 |
| MaritalStatus | 0 |
| MonthlyIncome | 0 |
| MonthlyRate | 0 |
| NumCompaniesWorked | 0 |
| Over18 | 0 |
| OverTime | 0 |
| PercentSalaryHike | 0 |
| PerformanceRating | 0 |
| RelationshipSatisfaction | 0 |
| StandardHours | 0 |
| StockOptionLevel | 0 |
| TotalWorkingYears | 0 |
| TrainingTimesLastYear | 0 |
| WorkLifeBalance | 0 |
| YearsAtCompany | 0 |
| YearsInCurrentRole | 0 |
| YearsSinceLastPromotion | 0 |
| YearsWithCurrManager | 0 |
| Attrition | 0 |

Selanjutnya, dilakukan **deteksi *outlier*** menggunakan metode **Interquartile Range (IQR)** untuk setiap fitur numerik.  
Hasil analisis menunjukkan bahwa beberapa variabel memiliki jumlah *outlier* yang cukup signifikan.  
Fitur **Attrition (190 *outlier*)**, **PerformanceRating (185 *outlier*)**, dan **TrainingTimesLastYear (174 *outlier*)** merupakan variabel dengan jumlah *outlier* terbanyak.  
Kondisi ini mengindikasikan adanya variasi ekstrem dalam tingkat performa, frekuensi pelatihan, serta status keluar atau bertahannya karyawan.  

Selain itu, fitur seperti **MonthlyIncome (86 *outlier*)**, **YearsSinceLastPromotion (85 *outlier*)**, dan **StockOptionLevel (66 *outlier*)** juga menunjukkan keberadaan *outlier* yang cukup tinggi.Hal ini mencerminkan adanya perbedaan besar dalam pendapatan, frekuensi promosi, dan kepemilikan saham antar karyawan.  

Beberapa variabel lain seperti **TotalWorkingYears (52 *outlier*)** dan **YearsAtCompany (52 *outlier*)** juga menunjukkan variasi signifikan dalam lama pengalaman kerja dan masa kerja di perusahaan. Sementara itu, fitur seperti **JobSatisfaction**, **JobLevel**, **JobInvolvement**, serta berbagai variabel demografis dan lingkungan kerja lainnya **tidak memiliki *outlier*** sama sekali, dimana hal ini menandakan distribusi nilai yang relatif seragam dan stabil di antara karyawan.

| Fitur | Jumlah Outlier |
|-------|----------------|
| Attrition | 190 |
| PerformanceRating | 185 |
| TrainingTimesLastYear | 174 |
| MonthlyIncome | 86 |
| YearsSinceLastPromotion | 85 |
| StockOptionLevel | 66 |
| TotalWorkingYears | 52 |
| YearsAtCompany | 52 |
| NumCompaniesWorked | 36 |
| YearsInCurrentRole | 16 |
| YearsWithCurrManager | 10 |
| JobSatisfaction | 0 |
| JobLevel | 0 |
| JobInvolvement | 0 |
| HourlyRate | 0 |
| EnvironmentSatisfaction | 0 |
| EmployeeNumber | 0 |
| EmployeeCount | 0 |
| Education | 0 |
| DistanceFromHome | 0 |
| DailyRate | 0 |
| Age | 0 |
| PercentSalaryHike | 0 |
| StandardHours | 0 |
| RelationshipSatisfaction | 0 |
| MonthlyRate | 0 |
| WorkLifeBalance | 0 |

### [Exploratory Data Analysis] - Univariate Analysis
#### Grafik 1 : Distribusi Kategori Employee Attrition
![Distribusi Kategori Employee-Attrition](https://github.com/natashavirnaa/employee-attrition-prediction/blob/main/image/EDA%201%20-%20Distribusi%20Kategori%20Employee-Attrition.png?raw=true)

Distribusi variabel **Attrition** menunjukkan bahwa sebagian besar karyawan **tidak keluar dari perusahaan (sekitar 84%)**, sedangkan hanya sekitar **16% yang mengalami attrition (keluar)**.  
Kondisi ini **menandakan adanya ketidakseimbangan kelas** yang cukup besar antara karyawan yang bertahan dan yang keluar, yang dapat **mempengaruhi performa model prediksi**.  

Untuk mengatasi hal tersebut, dilakukan beberapa strategi berikut:
1. **Stratified Train-Test Split**  
   Pembagian data dilakukan secara *terstratifikasi* agar proporsi antara karyawan yang keluar dan tidak keluar tetap konsisten di data pelatihan dan pengujian.  
2. **Penyesuaian Bobot Kelas (*Class Weight Adjustment*)**  
   Model diberikan bobot lebih besar pada kelas minoritas (**Attrition = 1**) agar kesalahan prediksi terhadap karyawan yang keluar tidak diabaikan.  
3. **Evaluasi dengan Metrik Sensitif terhadap Ketidakseimbangan**  
   Penggunaan metrik seperti **Recall**, *F1-Score*, dan **ROC-AUC** difokuskan agar model tidak hanya akurat pada kelas mayoritas, tetapi juga mampu mengenali pola karyawan yang berpotensi keluar.






