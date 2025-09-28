from flask import Flask, render_template, request
from serpapi import GoogleSearch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import os

app = Flask(__name__)

# 🔑 API Keys
SERPAPI_KEY = "8a4bdacd0dd9c07a9db3db7a65c16c58329b1ee99acd71f31435e6efc916d032"
HF_TOKEN = "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"

# ✅ Load IndoBERT sentiment model sekali di awal
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ✅ Load spaCy multilingual model (punya tokenisasi + POS tagging)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'xx_ent_wiki_sm' tidak ditemukan.\n"
        "Jalankan: python -m spacy download xx_ent_wiki_sm"
    )


# 🔧 Fungsi untuk ekstraksi noun & noun phrase
def extract_nouns_phrases(texts):
    """Ekstraksi nouns & noun phrases (heuristik kalau noun_chunks tidak tersedia)."""
    nouns = []
    noun_phrases = []

    for doc in nlp.pipe(texts, disable=["ner"]):
        # Kumpulkan nouns/proper nouns
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                lemma = token.lemma_.lower().strip()
                if lemma:
                    nouns.append(lemma)

        # Kalau noun_chunks tidak tersedia, pakai heuristik
        if hasattr(doc, "noun_chunks") and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                if phrase:
                    noun_phrases.append(phrase)
        else:
            cur = []
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                    cur.append(token.text)
                else:
                    if cur:
                        noun_phrases.append(" ".join(cur).lower())
                        cur = []
            if cur:
                noun_phrases.append(" ".join(cur).lower())

    return nouns, noun_phrases


# 🔧 Analisis aspek (frequent nouns & noun phrases per sentiment)
def aspect_analysis(reviews_by_sentiment):
    aspect_result = {}

    for sentiment, texts in reviews_by_sentiment.items():
        if not texts:
            aspect_result[sentiment] = {"nouns": [], "phrases": []}
            continue

        nouns, noun_phrases = extract_nouns_phrases(texts)

        aspect_result[sentiment] = {
            "nouns": Counter(nouns).most_common(20),
            "phrases": Counter(noun_phrases).most_common(20),
        }

    return aspect_result


# 🔧 Buat WordCloud dari semua review
def generate_wordcloud(all_texts, output_path="static/wordcloud.png"):
    if not all_texts:
        return None

    text_combined = " ".join(all_texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_combined)
    wc.to_file(output_path)
    return output_path


@app.route("/", methods=["GET", "POST"])
def index():
    analyzed_reviews = []
    raw_json = {}
    place_id = None
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_result = {"positive": {}, "negative": {}, "neutral": {}}
    wordcloud_path = None

    if request.method == "POST":
        product_name = request.form.get("product", "").strip()

        if not product_name:
            return render_template("index.html",
                                   reviews=[],
                                   raw_json={"error": "Form input kosong"},
                                   place_id=None,
                                   sentiment_summary=sentiment_summary,
                                   aspect_result=aspect_result,
                                   wordcloud_path=None)

        print(f"🔎 Cari produk: {product_name}")

        # 1️⃣ Cari place_id di Google Maps
        search = GoogleSearch({
            "engine": "google_maps",
            "q": product_name,
            "type": "search",
            "api_key": SERPAPI_KEY
        })
        maps_results = search.get_dict()
        raw_json = maps_results

        if "place_results" in maps_results:
            place_id = maps_results["place_results"].get("place_id")
        elif "local_results" in maps_results and len(maps_results["local_results"]) > 0:
            place_id = maps_results["local_results"][0].get("place_id")

        print("🔎 place_id ditemukan:", place_id)

        if place_id:
            # 2️⃣ Ambil review
            review_search = GoogleSearch({
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY
            })
            results = review_search.get_dict()
            raw_json = results
            reviews = results.get("reviews", [])

            print("🔎 Jumlah review berhasil diambil:", len(reviews))

            # 3️⃣ Analisis sentimen batch
            texts = [r.get("snippet", "")[:512] for r in reviews if r.get("snippet", "").strip()]
            if texts:
                sentiments = sentiment_pipeline(texts, batch_size=8)
                for r, sentiment in zip(reviews, sentiments):
                    text = r.get("snippet", "")
                    label = sentiment["label"].lower()
                    if label not in sentiment_summary:
                        label = "neutral"
                    sentiment_summary[label] += 1
                    analyzed_reviews.append({
                        "user": r["user"]["name"],
                        "text": text,
                        "sentiment": label,
                        "score": round(sentiment["score"], 3)
                    })

            print("📊 Ringkasan sentimen:", sentiment_summary)

            # 4️⃣ Aspek Analisis per Sentimen
            reviews_by_sentiment = {"positive": [], "negative": [], "neutral": []}
            for r in analyzed_reviews:
                reviews_by_sentiment[r["sentiment"]].append(r["text"])

            aspect_result = aspect_analysis(reviews_by_sentiment)

            # 5️⃣ WordCloud dari semua review
            all_texts = [r["text"] for r in analyzed_reviews]
            wordcloud_path = generate_wordcloud(all_texts)

        else:
            print("⚠️ place_id tetap tidak ditemukan")

    return render_template("index2.html",
                           reviews=analyzed_reviews,
                           raw_json=raw_json,
                           place_id=place_id,
                           sentiment_summary=sentiment_summary,
                           aspect_result=aspect_result,
                           wordcloud_path=wordcloud_path)


if __name__ == "__main__":
    app.run(debug=True)
