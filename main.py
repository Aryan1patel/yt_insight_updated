import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Models
# -----------------------------

class CommentsRequest(BaseModel):
    comments: list[str]


class TimestampComment(BaseModel):
    text: str
    timestamp: str


class TimestampRequest(BaseModel):
    comments: list[TimestampComment]


class SentimentCountRequest(BaseModel):
    sentiment_counts: dict


class WordcloudRequest(BaseModel):
    comments: list[str]


class TrendRequest(BaseModel):
    sentiment_data: list


# -----------------------------
# Preprocessing
# -----------------------------

stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
lemmatizer = WordNetLemmatizer()

# def preprocess_comment(comment):
    # try:
    #     comment = comment.lower().strip()
    #     comment = re.sub(r"\n", " ", comment)
    #     comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    #     stop_words = set(stopwords.words("english")) - {
    #         "not", "but", "however", "no", "yet"
    #     }

    #     comment = " ".join(
    #         [word for word in comment.split() if word not in stop_words]
    #     )

    #     lemmatizer = WordNetLemmatizer()

    #     comment = " ".join(
    #         [lemmatizer.lemmatize(word) for word in comment.split()]
    #     )

    #     return comment

    # except Exception as e:
    #     print(e)
    #     return comment

def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        comment = " ".join(
            [word for word in comment.split() if word not in stop_words]
        )

        comment = " ".join(
            [lemmatizer.lemmatize(word) for word in comment.split()]
        )

        return comment

    except Exception as e:
        print(e)
        return comment


# -----------------------------
# Load Model
# -----------------------------

def load_model(model_path, vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model(
    "./lgbm_model.pkl",
    "./tfidf_vectorizer.pkl"
)

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI sentiment service"}


# -----------------------------
# Prediction API
# -----------------------------

@app.post("/predict")
def predict(data: CommentsRequest):

    comments = data.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed = [preprocess_comment(c) for c in comments]

        # transformed = vectorizer.transform(preprocessed)
        # dense = transformed.toarray()

        # predictions = model.predict(dense).tolist()

        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed).tolist()

        response = [
            {"comment": c, "sentiment": s}
            for c, s in zip(comments, predictions)
        ]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Prediction with timestamps
# -----------------------------

@app.post("/predict_with_timestamps")
def predict_with_timestamps(data: TimestampRequest):

    comments = [item.text for item in data.comments]
    timestamps = [item.timestamp for item in data.comments]

    try:
        preprocessed = [preprocess_comment(c) for c in comments]

        # transformed = vectorizer.transform(preprocessed)
        # dense = transformed.toarray()

        # predictions = model.predict(dense).tolist()

        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed).tolist()

        response = [
            {
                "comment": c,
                "sentiment": str(s),
                "timestamp": t
            }
            for c, s, t in zip(comments, predictions, timestamps)
        ]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Pie Chart
# -----------------------------

@app.post("/generate_chart")
def generate_chart(data: SentimentCountRequest):

    sentiment_counts = data.sentiment_counts

    labels = ["Positive", "Neutral", "Negative"]

    sizes = [
        int(sentiment_counts.get("1", 0)),
        int(sentiment_counts.get("0", 0)),
        int(sentiment_counts.get("-1", 0))
    ]

    if sum(sizes) == 0:
        raise HTTPException(400, "Sentiment counts sum to zero")

    colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "white"}
    )

    plt.axis("equal")

    img = io.BytesIO()
    plt.savefig(img, format="png", transparent=True)
    img.seek(0)
    plt.close()

    return StreamingResponse(img, media_type="image/png")


# -----------------------------
# Wordcloud
# -----------------------------

@app.post("/generate_wordcloud")
def generate_wordcloud(data: WordcloudRequest):

    comments = data.comments

    preprocessed = [preprocess_comment(c) for c in comments]

    text = " ".join(preprocessed)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="Blues",
        stopwords=set(stopwords.words("english")),
        collocations=False,
    ).generate(text)

    img = io.BytesIO()
    wordcloud.to_image().save(img, format="PNG")
    img.seek(0)

    return StreamingResponse(img, media_type="image/png")


# -----------------------------
# Trend Graph
# -----------------------------

@app.post("/generate_trend_graph")
def generate_trend_graph(data: TrendRequest):

    df = pd.DataFrame(data.sentiment_data)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)

    df["sentiment"] = df["sentiment"].astype(int)

    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)

    monthly_totals = monthly_counts.sum(axis=1)

    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

    for s in [-1, 0, 1]:
        if s not in monthly_percentages.columns:
            monthly_percentages[s] = 0

    monthly_percentages = monthly_percentages[[-1, 0, 1]]

    plt.figure(figsize=(12, 6))

    colors = {-1: "red", 0: "gray", 1: "green"}

    for s in [-1, 0, 1]:
        plt.plot(
            monthly_percentages.index,
            monthly_percentages[s],
            marker="o",
            label=sentiment_labels[s],
            color=colors[s],
        )

    plt.title("Monthly Sentiment Percentage Over Time")
    plt.xlabel("Month")
    plt.ylabel("Percentage")
    plt.grid(True)

    plt.xticks(rotation=45)

    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return StreamingResponse(img, media_type="image/png")