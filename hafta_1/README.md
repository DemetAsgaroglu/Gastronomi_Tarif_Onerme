# Gastronomi Tarif Önerme Projesi
Elinizdeki proje, gastronomi tariflerini analiz etmek, işlemek ve önerilerde bulunmak için geliştirilmiş bir Python projesidir. Proje, metin işleme, TF-IDF hesaplama, Word2Vec model eğitimi ve Zipf yasası gibi çeşitli doğal dil işleme (NLP) tekniklerini içermektedir.
---

## Projenin Amacı

Bu projenin temel amacı, yemek tariflerini daha anlaşılır ve işlenebilir bir hale getirmek ve bu tariflerden anlamlı bilgiler çıkararak kullanıcıya önerilerde bulunmaktır. Proje kapsamında şu işlemler gerçekleştirilir:

1. **Veri Ön İşleme**: Tariflerdeki malzeme miktarları, birimler ve sıcaklık değerleri gibi bilgilerin ayrıştırılması ve temizlenmesi.
2. **Metin Analizi**: Tariflerin metin tabanlı analizinin yapılması (örneğin, TF-IDF ve Word2Vec yöntemleriyle).
4. **Görselleştirme**: Tariflerdeki verilerin görselleştirilmesi ve analiz sonuçlarının sunulması.

---

## KULLANILAN VERİ SETİNİN TANITIMI

Bu çalışmada, yemek tarifleriyle ilgili doğal dil işleme temelli bir öneri sistemi geliştirmek amacıyla, Kaggle üzerinden erişilebilen “Better Recipes for a Better Life” adlı veri seti kullanılmıştır. 
Boyut: 1.74 MB 
İçinde iki .csv dosyası vardır.  “ recipes.csv ” ve “ test_recipes.csv ” olmak üzere iki dosyadır. Biz  “ recipes.csv ” kullandık. 
Kaggle veri seti linki:
https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life

```

---

## Proje Yapısı ve Dosyaların İçeriği

Proje dosyaları ve klasörleri aşağıdaki gibi organize edilmiştir:

```
Gastronomi_Tarif_Onerme/
├── hafta_1/
│   ├── .idea/                 # IDE yapılandırma dosyaları
│   ├── data/                  # Veri setleri ve işlenmiş veriler
│   │   ├── recipes_cleaned.csv
│   │   ├── tfidf_lemmatized_results.csv
│   │   ├── tfidf_stemmed_results.csv
│   │   └── veriseti/          # Ham veri setleri
│   │       └── recipes.csv
│   ├── gorsel/                # Görselleştirme sonuçları
│   │   ├── tf-ıdf/            # TF-IDF görselleştirme sonuçları
│   │   └── word2vec/          # Word2Vec görselleştirme sonuçları
│   ├── model/                 # Eğitimli modeller
│   ├── src/                   # Kaynak kodlar
│   │   ├── gastronomi_preprocessing.py  # Veri ön işleme kodları
│   │   ├── tf-ıdf.py                   # TF-IDF hesaplama kodları
│   │   ├── word2vec.py                 # Word2Vec model eğitimi
│   │   └── zipf_graph.py               # Zipf Yasası analizi
│   └── Zipf Yasası/           # Zipf Yasası ile ilgili çalışmalar
└── README.md                  # Proje açıklama dosyası
```

## Kurulum

   ```

1. **NLTK modüllerini indirin**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Veri setlerini yerleştirin**:
   Ham veri setlerini `data/veriseti/` klasörüne yerleştirin.

---


## Kullanılan Teknolojiler ve Kütüphaneler

Proje, aşağıdaki teknolojiler ve Python kütüphaneleri kullanılarak geliştirilmiştir:

- **Python**: Projenin temel programlama dili.
- **Pandas**: Veri işleme ve analiz için.
- **NumPy**: Sayısal hesaplamalar için.
- **NLTK**: Doğal dil işleme işlemleri için.
- **Matplotlib**: Veri görselleştirme için.
- **Scikit-learn**: TF-IDF hesaplama ve makine öğrenimi işlemleri için.
- **Word2Vec (Gensim)**: Kelime vektörleri oluşturmak için.

---
### 1. `src/gastronomi_preprocessing.py`
Bu dosya, tariflerin metin işleme adımlarını gerçekleştiren fonksiyonları içerir. Tariflerdeki malzeme miktarları, birimler ve sıcaklık değerleri regex kullanılarak ayrıştırılır ve temizlenir.

#### Öne Çıkan Fonksiyonlar:
- **`extract_quantities_units_and_temperatures(directions)`**:
  - Tariflerdeki sayısal değerleri, birimleri ve sıcaklık bilgilerini ayrıştırır.
  - Çıktı: `quantities`, `units` ve `temperatures` sütunları.

- **`remove_quantities_units_and_temperatures(directions)`**:
  - Tariflerdeki sayısal değerleri, birimleri ve sıcaklık bilgilerini temizler.

- **`preprocess_sentence(sentence)`**:
  - Bir cümleyi kelimelere ayırır, stopword'leri çıkarır, lemmatize ve stemleme işlemlerini uygular.

- **`clean_and_process_directions(df)`**:
  - Tariflerin `directions` sütununu temizler, lemmatize ve stemleme işlemlerini uygular.

#### Çıktılar:
- İşlenmiş tarifler `data/recipes_cleaned.csv` dosyasına kaydedilir.

---

### 2. `src/tf-ıdf.py`
Bu dosya, tariflerin TF-IDF (Term Frequency-Inverse Document Frequency) değerlerini hesaplar ve görselleştirir.

#### Öne Çıkan Fonksiyonlar:
- **`calculate_tfidf(df, column_name)`**:
  - Belirtilen sütundaki metinler için TF-IDF değerlerini hesaplar.
  - Çıktı: TF-IDF matrisleri ve özellik isimleri.

- **`print_top_10_and_plot(tfidf_df, features, name)`**:
  - TF-IDF değerlerine göre en yüksek 10 kelimeyi terminale yazdırır ve bir bar grafiği oluşturur.

#### Çıktılar:
- TF-IDF sonuçları `data/tfidf_lemmatized_results.csv` ve `data/tfidf_stemmed_results.csv` dosyalarına kaydedilir.
- Görselleştirme sonuçları `gorsel/tf-ıdf/` klasörüne kaydedilir.

---

### 3. `src/word2vec.py`
Bu dosya, Word2Vec modellerini eğitir ve tariflerin kelime vektörlerini analiz eder.

#### Öne Çıkan Fonksiyonlar:
- **`train_and_save_model(corpus, params, model_name_prefix)`**:
  - Belirtilen parametrelerle Word2Vec modeli eğitir ve kaydeder.

- **`plot_word_vectors(model, title, filename)`**:
  - Eğitilen modellerin kelime vektörlerini PCA ile 2 boyuta indirger ve görselleştirir.

#### Çıktılar:
- Word2Vec modelleri `model/` klasörüne kaydedilir.
- Görselleştirme sonuçları `gorsel/word2vec/` klasörüne kaydedilir.

---

### 4. `src/zipf_graph.py`
Bu dosya, Zipf yasasını analiz etmek için grafikler oluşturur.

#### Öne Çıkan Fonksiyonlar:
- **`plot_zipf(word_freq, title, filename)`**:
  - Kelime frekanslarını kullanarak Zipf grafiği çizer ve kaydeder.

#### Çıktılar:
- Zipf grafikleri `Zipf Yasası/` klasörüne kaydedilir.

---

### 5. Veri Dosyaları (`data/`)
- **`recipes_cleaned.csv`**:
  - Temizlenmiş ve işlenmiş tarif verilerini içerir.
- **`tfidf_lemmatized_results.csv` ve `tfidf_stemmed_results.csv`**:
  - TF-IDF hesaplama sonuçlarını içerir.
- **`recipes.csv`**:
  - Ham tarif verilerini içerir.

---

### 6. Model Dosyaları (`model/`)
- **`lemmatized_model_cbow_window2_dim100.model`** gibi dosyalar:
  - Word2Vec modellerini içerir. Bu modeller, tariflerin lemmatize edilmiş veya stemlenmiş versiyonları üzerinde eğitilmiştir.

---

### 7. Görselleştirme Dosyaları (`gorsel/`)
- **`tf-ıdf/`**:
  - TF-IDF görselleştirme sonuçlarını içerir.
- **`word2vec/`**:
  - Word2Vec görselleştirme sonuçlarını içerir.

---
## Örnek Çıktılar

### TF-IDF En Yüksek 10 Kelime
![TF-IDF Görselleştirme](hafta_1/gorsel/tf-ıdf/top_10_tfidf.png)

### Word2Vec PCA Görselleştirme
![Word2Vec Görselleştirme](hafta_1/gorsel/word2vec/pca_visualization.png)

### Zipf Grafiği
![Zipf Grafiği](hafta_1/Zipf Yasası/zipf_raw.png)
