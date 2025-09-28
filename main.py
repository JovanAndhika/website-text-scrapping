import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import spacy
from wordcloud import WordCloud
import io
import base64
from collections import Counter

# --- Inisialisasi Aplikasi dan Model ---
app = Flask(__name__)
app.secret_key = 'kunci_rahasia_anda'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memuat model spaCy untuk NLP (dijalankan sekali saat aplikasi start)
try:
    nlp = spacy.load("en_core_web_sm")
    print("Model spaCy 'en_core_web_sm' berhasil dimuat.")
except OSError:
    print("Model spaCy 'en_core_web_sm' tidak ditemukan.")
    print("Jalankan: python -m spacy download en_core_web_sm")
    nlp = None

# Memuat pipeline Hugging Face untuk analisis sentimen
from transformers import pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)
print("Pipeline analisis sentimen berhasil dimuat.")

# --- Fungsi Helper ---
def allowed_file(filename):
    """Mengecek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_wordcloud(text):
    """Membuat gambar word cloud dan mengembalikannya sebagai string base64."""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_b64

def extract_key_phrases(text, nlp_model):
    """Mengekstrak nouns dan noun phrases menggunakan spaCy."""
    if not nlp_model:
        return [], []
        
    doc = nlp_model(text)
    nouns = [
        token.lemma_.lower() for token in doc 
        if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 2
    ]
    noun_phrases = [
        chunk.text.lower() for chunk in doc.noun_chunks 
        if len(chunk.text.split()) > 1 and not any(token.is_stop for token in chunk)
    ]
    return nouns, noun_phrases

# --- Routes Aplikasi ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Tidak ada bagian file')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df.dropna(how='all', inplace=True)
            df.to_csv(filepath, index=False)
            table_html = df.head().to_html(classes='min-w-full divide-y divide-gray-200', border=0)
            return render_template('result.html', filename=file.filename, table=table_html)
        except Exception as e:
            flash(f"Terjadi error saat memproses file: {e}")
            return redirect(request.url)
            
    flash('Jenis file tidak diizinkan')
    return redirect(request.url)


@app.route('/files')
def list_files():
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
    return render_template('files.html', files=files)



@app.route('/analyze/<filename>')
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("File tidak ditemukan.")
        return redirect(url_for('list_files'))

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if 'Ulasan' not in df.columns:
        flash("File CSV tidak memiliki kolom 'Ulasan'.")
        return redirect(url_for('list_files'))
        
    # SAMPLE DF akan menampilkan 100 ulasan teratas
    sample_df = df.head(100).copy()
    reviews = sample_df['Ulasan'].dropna().astype(str).tolist()
    
    if not reviews:
        flash("Tidak ada ulasan untuk dianalisis di dalam file.")
        return redirect(url_for('list_files'))

    # 1. Analisis Sentimen
    sentiment_results = sentiment_pipeline(reviews)
    sample_df['Sentimen'] = [res['label'] for res in sentiment_results]
    sample_df['Skor Sentimen'] = [round(res['score'], 4) for res in sentiment_results]
    
    # 2. Ringkasan Sentimen
    sentiment_counts = sample_df['Sentimen'].value_counts()
    summary = {
        'positive': int(sentiment_counts.get('positive', 0)),
        'negative': int(sentiment_counts.get('negative', 0)),
        'neutral': int(sentiment_counts.get('neutral', 0)),
    }
    chart_labels = list(summary.keys())
    chart_data = list(summary.values())

    # 3. Ekstraksi Frasa & Word Cloud
    all_reviews_text = " ".join(reviews)
    nouns, noun_phrases = extract_key_phrases(all_reviews_text, nlp)
    
    # Hitung frekuensi INI UNTUK MENGATUR BANYAKNYA NOUNS YANG INGIN DITAMPILKAN
    top_nouns = Counter(nouns).most_common(10)
    top_noun_phrases = Counter(noun_phrases).most_common(10)
    
    # Persiapkan data untuk diagram baru
    noun_labels, noun_data = zip(*top_nouns) if top_nouns else ([], [])
    phrase_labels, phrase_data = zip(*top_noun_phrases) if top_noun_phrases else ([], [])
    
    wordcloud_text = " ".join(nouns + noun_phrases)
    wordcloud_image = generate_wordcloud(wordcloud_text) if wordcloud_text else None
    
    table_html = sample_df.to_html(classes='min-w-full divide-y divide-gray-200', border=0, index=False)
    
    return render_template(
        'analysis_result.html',
        filename=filename,
        summary=summary,
        chart_labels=chart_labels,
        chart_data=chart_data,
        table=table_html,
        wordcloud_image=wordcloud_image,
        # Data baru untuk diagram batang dan pie
        noun_labels=list(noun_labels),
        noun_data=list(noun_data),
        phrase_labels=list(phrase_labels),
        phrase_data=list(phrase_data)
    )

if __name__ == '__main__':
    app.run(debug=True)

