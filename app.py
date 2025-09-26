import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from transformers import pipeline

# Load summarization model (small + free on HuggingFace)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="GitHub Repo Insights + AI", layout="wide")

st.title("📊 GitHub Repository Insights + 🤖 AI Summary")

repo = st.text_input("Enter repository (owner/repo)", "streamlit/streamlit")

if repo:
    url = f"https://api.github.com/repos/{repo}"
    r = requests.get(url).json()

    if "message" in r:
        st.error("Repo not found or API limit reached.")
    else:
        st.subheader(f"📦 {r['full_name']}")
        st.write(r["description"])
        st.metric("⭐ Stars", r["stargazers_count"])
        st.metric("🍴 Forks", r["forks_count"])
        st.metric("🐛 Open Issues", r["open_issues_count"])

        # Contributors
        contrib_url = f"https://api.github.com/repos/{repo}/contributors"
        contrib_data = requests.get(contrib_url).json()
        if isinstance(contrib_data, list):
            df = pd.DataFrame(contrib_data)[["login", "contributions"]]
            fig = px.bar(df, x="login", y="contributions", title="Top Contributors")
            st.plotly_chart(fig, use_container_width=True)

        # AI Summary
        st.subheader("🤖 AI-Generated Repo Summary")
        text_to_summarize = (r["description"] or "") + " " + \
                            f"This repository has {r['stargazers_count']} stars, {r['forks_count']} forks, and {r['open_issues_count']} open issues."
        if text_to_summarize.strip():
            summary = summarizer(text_to_summarize, max_length=60, min_length=20, do_sample=False)
            st.info(summary[0]['summary_text'])
