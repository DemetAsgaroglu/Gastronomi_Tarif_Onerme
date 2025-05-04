import os
import time
from gensim.models import Word2Vec
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Temizlenmiş veri dosyasını yükle
df = pd.read_csv("data/recipes_cleaned.csv", converters={
    "lemmatized_directions": eval,
    "stemmed_directions": eval
})

# Word2Vec için corpus listesi
tokenized_corpus_lemmatized = df["lemmatized_directions"].tolist()
tokenized_corpus_stemmed = df["stemmed_directions"].tolist()

# Model parametre kombinasyonları
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Model eğitimi ve kaydetme fonksiyonu (süreli)
def train_and_save_model(corpus, params, model_name_prefix):
    start_time = time.time()

    model = Word2Vec(
        sentences=corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,
        sg=1 if params['model_type'] == 'skipgram' else 0
    )

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"{model_name_prefix}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model"
    model_path = os.path.join(model_dir, model_name)

    model.save(model_path)

    end_time = time.time()
    duration = end_time - start_time
    print(f"{model_name} saved! Training time: {duration:.2f} seconds")

# Lemmatize edilmiş verilerle model eğitimi
print("\n--- Training models on lemmatized corpus ---\n")
for param in parameters:
    train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model")

# Stemlenmiş verilerle model eğitimi
print("\n--- Training models on stemmed corpus ---\n")
for param in parameters:
    train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model")


model_paths = {
    'lemma_cbow_w2_d100': 'model/lemmatized_model_cbow_window2_dim100.model',
    'stem_skipgram_w4_d300': 'model/stemmed_model_skipgram_window4_dim300.model',
    'lemma_skipgram_w2_d100': 'model/lemmatized_model_skipgram_window2_dim100.model'
}

# Hedef kelime
target_word = "chicken"

# PCA işlemini her model için ayrı yapacağız
for model_name, model_path in model_paths.items():
    model = Word2Vec.load(model_path)

    if target_word not in model.wv:
        print(f"{target_word} not in vocabulary for {model_name}")
        continue

    # En benzer kelimeleri al
    similar_words = model.wv.most_similar(target_word, topn=30)

    # İlk 5 kelimeyi terminale yazdır
    print(f"\nModel: {model_name}")
    print(f"En benzer 5 kelime:")
    for word, similarity in similar_words[:5]:
        print(f"{word}: {similarity:.4f}")

    # Hedef kelimeyi ve benzer kelimeleri vektörler ile toplamak
    all_words = [(model_name, target_word)]
    vectors = [model.wv[target_word]]
    for word, _ in similar_words:
        if word in model.wv:
            all_words.append((model_name, word))
            vectors.append(model.wv[word])

    # PCA ile 2 boyuta indirgeme
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    # Görselleştirme
    plt.figure(figsize=(8, 6))
    plt.title(f"PCA Görselleştirme: '{model_name}' için '{target_word}' ve Benzer Kelimeler")
    for i, (word) in enumerate(all_words):
        plt.scatter(result[i, 0], result[i, 1], label=word[1], alpha=0.7)
        plt.text(result[i, 0] + 0.01, result[i, 1] + 0.01, word[1], fontsize=9)

    # Etiketleme ve stil
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)

    # Klasör oluşturup görseli kaydetme
    os.makedirs("gorsel/word2vec", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"gorsel/word2vec/{model_name}_pca_comparison.png")
    plt.close()