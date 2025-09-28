from flask import Flask, render_template, request
from serpapi import GoogleSearch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import Counter
from wordcloud import WordCloud
import spacy
import os

app = Flask(__name__)

#  API Key SerpAPI (ganti dengan punyamu)
SERPAPI_KEY = "8a4bdacd0dd9c07a9db3db7a65c16c58329b1ee99acd71f31435e6efc916d032"

#  Token Hugging Face (ganti dengan punyamu jika model butuh akses)
HF_TOKEN = "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"

#  Load IndoBERT sentiment model sekali di awal
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#  Load spaCy English model (lebih stabil untuk noun_chunks)
nlp = spacy.load("en_core_web_sm")


# ==========================================
# Fungsi bantu
# ==========================================
def extract_nouns_phrases(texts):
    """Ekstrak kata benda & frasa benda dari kumpulan teks"""
    nouns = []
    noun_phrases = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                nouns.append(token.lemma_.lower())
        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text.lower())
    return nouns, noun_phrases


def aspect_analysis(reviews_by_sentiment):
    """Analisis frequent nouns & noun phrases untuk setiap kategori sentimen"""
    aspect_result = {}
    for sentiment, texts in reviews_by_sentiment.items():
        nouns, noun_phrases = extract_nouns_phrases(texts)
        aspect_result[sentiment] = {
            "nouns": Counter(nouns).most_common(20),
            "phrases": Counter(noun_phrases).most_common(20),
        }
    return aspect_result


def generate_wordcloud(texts, output_path):
    """Generate wordcloud dari kumpulan teks"""
    if not texts:
        return None
    text = " ".join(texts)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(text)
    wc.to_file(output_path)
    return "/" + output_path


# ==========================================
# Routes
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    reviews = []
    analyzed_reviews = []
    raw_json = {}
    place_id = None
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_result = {}
    wordclouds = {"all": None, "positive": None, "negative": None, "neutral": None}

    if request.method == "POST":
        product_name = request.form.get("product", "").strip()

        if not product_name:
            return render_template("index.html",
                                   reviews=[],
                                   raw_json={"error": "Form input kosong"},
                                   place_id=None,
                                   sentiment_summary=sentiment_summary,
                                   aspect_result=aspect_result,
                                   wordclouds=wordclouds)

        print(f"Cari produk: {product_name}")

        # 1️ Cari place_id dulu dengan Google Maps
        search = GoogleSearch({
            "engine": "google_maps",
            "q": product_name,
            "type": "search",
            "api_key": SERPAPI_KEY
        })
        maps_results = search.get_dict()
        raw_json = maps_results  # simpan buat debug awal

        #  Ambil place_id dari beberapa kemungkinan field
        if "place_results" in maps_results:
            place_id = maps_results["place_results"].get("place_id")
        elif "local_results" in maps_results and len(maps_results["local_results"]) > 0:
            place_id = maps_results["local_results"][0].get("place_id")

        print("place_id ditemukan:", place_id)

        if place_id:
            # 2️ Ambil review dengan google_maps_reviews
            review_search = GoogleSearch({
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY
            })
            results = review_search.get_dict()
            raw_json = results  # overwrite untuk debug

            reviews = results.get("reviews", [])
            print("Jumlah review berhasil diambil:", len(reviews))

            # 3️ Analisis sentimen setiap review
            if reviews:
                for r in reviews:
                    text = r.get("snippet", "")
                    if text.strip():
                        sentiment = sentiment_pipeline(text[:512])[0]
                        label = sentiment["label"].lower()
                        if label not in sentiment_summary:
                            label = "neutral"  # fallback
                        sentiment_summary[label] += 1
                        analyzed_reviews.append({
                            "user": r["user"]["name"],
                            "text": text,
                            "sentiment": label,
                            "score": round(sentiment["score"], 3)
                        })

                print("Ringkasan sentimen:", sentiment_summary)

                # 4️ Aspek Analisis per Sentimen
                reviews_by_sentiment = {"positive": [], "negative": [], "neutral": []}
                for r in analyzed_reviews:
                    reviews_by_sentiment[r["sentiment"]].append(r["text"])

                aspect_result = aspect_analysis(reviews_by_sentiment)
                print("aspect_result:", aspect_result)

                # 5️ Generate Wordcloud dari semua review
                os.makedirs("static", exist_ok=True)
                texts_all = [r["text"] for r in analyzed_reviews]
                wordclouds["all"] = generate_wordcloud(texts_all, "static/wordcloud_all.png")

                # 6️ Generate Wordcloud per sentimen
                for sent in ["positive", "negative", "neutral"]:
                    wordclouds[sent] = generate_wordcloud(
                        reviews_by_sentiment[sent],
                        f"static/wordcloud_{sent}.png"
                    )

        else:
            print("place_id tetap tidak ditemukan")

    return render_template("index3.html",
                           reviews=analyzed_reviews,
                           raw_json=raw_json,
                           place_id=place_id,
                           sentiment_summary=sentiment_summary,
                           aspect_result=aspect_result,
                           wordclouds=wordclouds)


if __name__ == "__main__":
    # pastikan ada folder static
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
