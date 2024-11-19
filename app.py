import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Persiapan data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('skincare_products_clean.csv')

# Menghapus kolom 'product_url' dan 'price'
df = df.drop(columns=['product_url'])

# Gabungkan kolom untuk analisis
combined_df = df.astype(str).agg(' '.join, axis=1)

# Menghapus stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

removed_stopword = combined_df.apply(remove_stopwords)

# Tokenisasi Data
tokenized_data = removed_stopword.apply(word_tokenize)

# Menggunakan TF-IDF untuk mengubah teks menjadi vektor
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(removed_stopword)

# Simpan data vektor dan fitur untuk pencarian
tfidf_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())

# Fungsi untuk mencari produk berdasarkan query pengguna
def search_products(query):
    # Menghapus stopwords dari query
    cleaned_query = remove_stopwords(query)
    
    # Mengubah query menjadi vektor
    query_vector = vectorizer.transform([cleaned_query])
    
    # Menghitung cosine similarity antara query dan dataset
    cosine_similarities = cosine_similarity(query_vector, vectors)
    
    # Mengurutkan hasil pencarian berdasarkan cosine similarity
    sorted_indices = cosine_similarities[0].argsort()[::-1]
    
    # Menampilkan hasil pencarian berdasarkan urutan relevansi
    results = []
    for index in sorted_indices[:5]:  # Menampilkan 30 produk teratas
        product_data = df.iloc[index]
        relevance_score = cosine_similarities[0][index]
        results.append((product_data, relevance_score))
    
    return results, cosine_similarities, sorted_indices

# Halaman Input Search Keywords
if 'page' not in st.session_state:
    st.session_state.page = "Input Search Keywords"

def go_to_page(page):
    st.session_state.page = page

if st.session_state.page == "Input Search Keywords":
    st.title("SKINCARE SEARCH ENGINE")

    query = st.text_input("Masukkan kata kunci pencarian")

    if query:
        # Cari produk berdasarkan query
        search_results, cosine_similarities, sorted_indices = search_products(query)
        
        # Tampilkan hasil pencarian
        # Tampilkan hasil pencarian
        st.subheader("Hasil Pencarian:")
        for product_data, relevance in search_results:
            # Tampilkan nama produk beserta nilai relevansi
            st.write(f"Relevansi: {relevance:.4f}")
            if st.button(product_data['product_name'], key=product_data['product_name']):
                st.session_state.selected_product = product_data['product_name']
                go_to_page("Product View")

        # Evaluasi berdasarkan keseluruhan dataset
        threshold_recommend = 0.150
        threshold_ground_truth = 0.180
        
        # Menentukan relevansi berdasarkan threshold
        recommended_relevance = cosine_similarities[0][sorted_indices] >= threshold_recommend
        ground_truth_relevance = cosine_similarities[0][sorted_indices] >= threshold_ground_truth

        # Confusion matrix
        cm = confusion_matrix(ground_truth_relevance, recommended_relevance,
                              labels=[True, False])
        df_cm = pd.DataFrame(cm, range(2), range(2))

        fig = plt.figure(figsize=(10, 4))
        sns.set_theme(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, cmap='RdPu', fmt='g', cbar=False, annot_kws={"size": 16},
                    xticklabels=["True", "False"], yticklabels=["True", "False"])
        plt.xlabel("Prediction", fontsize=16, labelpad=20, fontweight="bold")
        plt.ylabel("Actual", rotation=0, fontsize=16, labelpad=50, fontweight="bold")
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix", fontsize=20, pad=25, fontweight="bold", x=0.4)

        # Menghitung TP, FP, FN
        tp = np.sum(np.logical_and(recommended_relevance, ground_truth_relevance))
        fp = np.sum(np.logical_and(recommended_relevance, ~ground_truth_relevance))
        fn = np.sum(np.logical_and(~recommended_relevance, ground_truth_relevance))

        # Menghitung Precision, Recall, dan F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Menampilkan metrik evaluasi untuk keseluruhan dataset
        st.subheader("Evaluasi berdasarkan Keseluruhan Dataset:")
        st.pyplot(fig)
        st.write(st.get_option('theme.backgroundColor'))
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-Score:** {f1:.4f}")
        
        # Evaluasi berdasarkan 5 teratas
        recommended_relevance_top = recommended_relevance[:5]
        ground_truth_relevance_top = ground_truth_relevance[:5]

        # Confusion matrix 5 teratas
        cm_top = confusion_matrix(ground_truth_relevance_top, recommended_relevance_top,
                                  labels=[True, False])
        df_cm_top = pd.DataFrame(cm_top, range(2), range(2))

        fig_top = plt.figure(figsize=(10, 4))
        sns.heatmap(df_cm_top, annot=True, cmap='RdPu', fmt='g', cbar=False, annot_kws={"size": 16},
                    xticklabels=["True", "False"], yticklabels=["True", "False"])
        plt.xlabel("Prediction", fontsize=16, labelpad=20, fontweight="bold")
        plt.ylabel("Actual", rotation=0, fontsize=16, labelpad=50, fontweight="bold")
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix", fontsize=20, pad=25, fontweight="bold", x=0.4)

        tp_top = np.sum(np.logical_and(recommended_relevance_top, ground_truth_relevance_top))  # True Positives
        fp_top = np.sum(np.logical_and(recommended_relevance_top, ~ground_truth_relevance_top))  # False Positives
        fn_top = np.sum(np.logical_and(~recommended_relevance_top, ground_truth_relevance_top))  # False Negatives

        # Menghitung Precision, Recall, dan F1-Score untuk 5 teratas
        precision_top = tp_top / (tp_top + fp_top) if (tp_top + fp_top) > 0 else 0
        recall_top = tp_top / (tp_top + fn_top) if (tp_top + fn_top) > 0 else 0
        f1_top = 2 * (precision_top * recall_top) / (precision_top + recall_top) if (precision_top + recall_top) > 0 else 0
        
        # Menampilkan metrik evaluasi untuk 5 teratas
        st.subheader("Evaluasi berdasarkan 5 Teratas:")
        st.pyplot(fig_top)
        st.write(f"**Precision:** {precision_top:.4f}")
        st.write(f"**Recall:** {recall_top:.4f}")
        st.write(f"**F1-Score:** {f1_top:.4f}")

elif st.session_state.page == "Product View":
    st.markdown("<div class='main detail-box'>", unsafe_allow_html=True)
    
    # Tampilkan Detail Produk
    selected_product = st.session_state.selected_product
    product_details = df[df['product_name'] == selected_product].iloc[0]

    st.markdown(f"### {product_details['product_name']}")
    st.write(f"**Type:** {product_details['product_type']}")
    st.write(f"**Ingredients:** {product_details['clean_ingreds']}")
    st.write(f"**Price:** {product_details['price']}")  # Sesuaikan kolom jika ada di dataset

    # Tombol untuk kembali ke halaman hasil pencarian
    if st.button("Kembali ke Hasil Pencarian", use_container_width=True):
        go_to_page("Input Search Keywords")
