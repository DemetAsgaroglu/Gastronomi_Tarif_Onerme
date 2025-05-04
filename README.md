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
