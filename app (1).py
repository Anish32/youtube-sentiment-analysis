import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import streamlit as st
import pandas as pd
from transformers import pipeline
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

# ✅ Replace this with your actual YouTube Data API key
API_KEY = "AIzaSyB1UqGu7hylBGE4Bwd09jKMDv8GYdQiR78"

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")
def get_video_comments(video_id, max_comments=50):
    comments = []
    nextPageToken = None
    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken,
                textFormat="plainText"
            ).execute()
        except Exception as e:
            # Check if it's the commentsDisabled error
            if "commentsDisabled" in str(e):
                raise ValueError("🚫 Comments are disabled on this video.")
            else:
                raise e

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break

        nextPageToken = response.get("nextPageToken")
        if not nextPageToken:
            break
    return comments


def analyze_comments(comments):
    results = sentiment_pipeline(comments)
    sentiments = [res["label"] for res in results]
    df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})
    return df

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

st.title("📊 YouTube Comment Sentiment Analyzer with Transformers")

video_url = st.text_input("Enter a YouTube Video URL:")
max_comments = st.slider("Max number of comments to fetch", 10, 200, 50)

if st.button("Analyze"):
    # Extract video ID robustly
    video_id = None
    if "v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]

    if not video_id:
        st.error("❌ Invalid YouTube URL.")
    else:
        with st.spinner("Fetching and analyzing comments..."):
            try:
                comments = get_video_comments(video_id, max_comments)

                if not comments:
                    st.warning("No comments found on this video.")
                else:
                    df = analyze_comments(comments)

                    # ✅ Save CSV to disk automatically
                    save_path = "sentiment_results_saved.csv"
                    df.to_csv(save_path, index=False)
                    st.success(f"✅ Results saved automatically to `{save_path}` in your working directory.")

                    st.subheader("📋 Sentiment Results")
                    st.dataframe(df)

                    st.write("### Positive Comments:")
                    st.dataframe(df[df["Sentiment"] == "POSITIVE"])

                    st.write("### Negative Comments:")
                    st.dataframe(df[df["Sentiment"] == "NEGATIVE"])

                    st.subheader("📈 Sentiment Distribution")
                    fig = px.pie(df, names="Sentiment", title="Sentiment Overview")
                    st.plotly_chart(fig)

                    st.subheader("☁️ Word Cloud")
                    generate_wordcloud(" ".join(df["Comment"]))

                    # ✅ Download button for convenience
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=df.to_csv(index=False).encode(),
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")
