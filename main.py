import os
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
from transformers import pipeline
import json # Diperlukan untuk Chart.js

# --- Inisialisasi Aplikasi dan Model ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Menentukan folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# Membuat folder 'uploads' jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi pipeline model HANYA SEKALI saat aplikasi dimulai
# Ini adalah praktik terbaik untuk efisiensi
try:
    print("Memuat model analisis sentimen...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    sentiment_pipeline = None

def allowed_file(filename):
    """Fungsi untuk memeriksa apakah ekstensi file diizinkan"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes Aplikasi ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada bagian file yang dipilih', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                df = pd.read_csv(filepath)
                df.dropna(how='all', inplace=True)
                df.to_csv(filepath, index=False)
                
                table_html = df.head().to_html(classes='min-w-full bg-white border border-gray-300', justify='left')
                return render_template('result.html', table=table_html, filename=filename)

            except Exception as e:
                flash(f'Terjadi error saat memproses file: {e}', 'danger')
                return redirect(request.url)

        else:
            flash('Jenis file tidak diizinkan. Harap unggah file CSV.', 'danger')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/files')
def list_files():
    """Route untuk menampilkan daftar file di folder uploads"""
    files_list = []
    try:
        files_list = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
    except FileNotFoundError:
        flash('Folder uploads tidak ditemukan.', 'danger')
    return render_template('files.html', files=files_list)


@app.route('/analyze/<filename>')
def analyze_sentiment(filename):
    """Route untuk menganalisis sentimen dari file yang dipilih"""
    if sentiment_pipeline is None:
        flash('Model analisis sentimen tidak berhasil dimuat. Silakan cek konsol server.', 'danger')
        return redirect(url_for('list_files'))

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Membaca data dan memastikan kolom 'Ulasan' ada
        df = pd.read_csv(filepath)
        # Menghapus spasi ekstra dari nama kolom (masalah umum)
        df.columns = df.columns.str.strip()
        
        if 'Ulasan' not in df.columns:
            flash(f"Error: File '{filename}' tidak memiliki kolom 'Ulasan'.", 'danger')
            return redirect(url_for('list_files'))

        # Mengambil ulasan, mengubah NaN menjadi string kosong
        reviews = df['Ulasan'].fillna('').astype(str).tolist()

        # Menjalankan analisis sentimen
        # Menggunakan batching untuk efisiensi jika data besar
        results = sentiment_pipeline(reviews, batch_size=16)

        # Menambahkan hasil ke dataframe
        df['Sentimen'] = [res['label'] for res in results]
        df['Skor Sentimen'] = [round(res['score'], 4) for res in results]

        # Menghitung ringkasan sentimen
        sentiment_counts = df['Sentimen'].value_counts().to_dict()
        summary = {
            'positive': sentiment_counts.get('positive', 0),
            'negative': sentiment_counts.get('negative', 0),
            'neutral': sentiment_counts.get('neutral', 0),
        }

        # Mengubah dataframe ke HTML untuk ditampilkan
        table_html = df.to_html(classes='min-w-full bg-white border border-gray-300', justify='left', index=False)
        
        return render_template('analysis_result.html', 
                               filename=filename, 
                               summary=summary,
                               chart_data=json.dumps(list(summary.values())),
                               chart_labels=json.dumps(list(summary.keys())),
                               table=table_html)

    except FileNotFoundError:
        flash(f"File '{filename}' tidak ditemukan.", 'danger')
        return redirect(url_for('list_files'))
    except Exception as e:
        flash(f"Terjadi error saat menganalisis file: {e}", 'danger')
        return redirect(url_for('list_files'))

if __name__ == '__main__':
    app.run(debug=True)

