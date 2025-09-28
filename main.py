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

# Memuat model spaCy untuk NLP
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_wordcloud(text, colormap='viridis'):
    """Membuat gambar word cloud dengan colormap yang bisa disesuaikan."""
    if not text or not text.strip():
        return None
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=colormap,
        max_words=100
    ).generate(text)
    
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_b64

def extract_key_phrases(text, nlp_model):
    if not nlp_model or not text or not text.strip():
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
        
    reviews = df['Ulasan'].dropna().astype(str).tolist()
    
    if not reviews:
        flash("Tidak ada ulasan untuk dianalisis di dalam file.")
        return redirect(url_for('list_files'))

    # 1. Analisis Sentimen
    sentiment_results = sentiment_pipeline(reviews)
    df['Sentimen'] = [res['label'] for res in sentiment_results]
    df['Skor Sentimen'] = [round(res['score'], 4) for res in sentiment_results]
    
    # 2. Ringkasan Sentimen
    sentiment_counts = df['Sentimen'].value_counts()
    summary = {
        'positive': int(sentiment_counts.get('positive', 0)),
        'negative': int(sentiment_counts.get('negative', 0)),
        'neutral': int(sentiment_counts.get('neutral', 0)),
    }
    chart_labels = list(summary.keys())
    chart_data = list(summary.values())

    # 3. Ekstraksi Frasa & Word Cloud per Sentimen
    positive_reviews_text = " ".join(df[df['Sentimen'] == 'positive']['Ulasan'].dropna().astype(str).tolist())
    negative_reviews_text = " ".join(df[df['Sentimen'] == 'negative']['Ulasan'].dropna().astype(str).tolist())
    neutral_reviews_text = " ".join(df[df['Sentimen'] == 'neutral']['Ulasan'].dropna().astype(str).tolist())
    
    pos_nouns, pos_phrases = extract_key_phrases(positive_reviews_text, nlp)
    neg_nouns, neg_phrases = extract_key_phrases(negative_reviews_text, nlp)
    neu_nouns, neu_phrases = extract_key_phrases(neutral_reviews_text, nlp)
    
    wc_positive = generate_wordcloud(" ".join(pos_nouns + pos_phrases), colormap='Greens')
    wc_negative = generate_wordcloud(" ".join(neg_nouns + neg_phrases), colormap='Reds')
    wc_neutral = generate_wordcloud(" ".join(neu_nouns + neu_phrases), colormap='Greys')

    # 4. Data untuk Diagram & Word Cloud Keseluruhan
    all_nouns = pos_nouns + neg_nouns + neu_nouns
    all_noun_phrases = pos_phrases + neg_phrases + neu_phrases
    
    # Menambahkan kembali word cloud umum
    wordcloud_text_overall = " ".join(all_nouns + all_noun_phrases)
    wc_overall = generate_wordcloud(wordcloud_text_overall, colormap='viridis')
    
    top_nouns = Counter(all_nouns).most_common(10)
    top_noun_phrases = Counter(all_noun_phrases).most_common(10)
    
    noun_labels, noun_data = zip(*top_nouns) if top_nouns else ([], [])
    phrase_labels, phrase_data = zip(*top_noun_phrases) if top_noun_phrases else ([], [])
    
    table_html = df.to_html(classes='min-w-full divide-y divide-gray-200', border=0, index=False)
    
    return render_template(
        'analysis_result.html',
        filename=filename,
        summary=summary,
        chart_labels=chart_labels,
        chart_data=chart_data,
        table=table_html,
        # Word clouds
        wordcloud_overall=wc_overall,
        wordcloud_positive=wc_positive,
        wordcloud_negative=wc_negative,
        wordcloud_neutral=wc_neutral,
        # Data untuk diagram
        noun_labels=list(noun_labels),
        noun_data=list(noun_data),
        phrase_labels=list(phrase_labels),
        phrase_data=list(phrase_data)
    )

if __name__ == '__main__':
    app.run(debug=True)

