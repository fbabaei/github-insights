import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="GitHub Repo Insights + AI", layout="wide")

st.title(" GitHub Repository Insights + AI Summary")

# Cache the summarizer so it loads once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

repo = st.text_input("Enter repository (owner/repo)", "streamlit/streamlit")

if repo:
    url = f"https://api.github.com/repos/{repo}"
    r = requests.get(url).json()

    if "message" in r:
        st.error("Repo not found or API limit reached.")
    else:
        st.subheader(f"{r['full_name']}")
        st.write(r["description"] or "No description available.")
        col1, col2, col3 = st.columns(3)
        col1.metric(" Stars", r["stargazers_count"])
        col2.metric(" Forks", r["forks_count"])
        col3.metric(" Open Issues", r["open_issues_count"])

        # Contributors
        contrib_url = f"https://api.github.com/repos/{repo}/contributors"
        contrib_data = requests.get(contrib_url).json()
        if isinstance(contrib_data, list) and len(contrib_data) > 0:
            df = pd.DataFrame(contrib_data)[["login", "contributions"]]
            fig = px.bar(df, x="login", y="contributions", title="Top Contributors")
            st.plotly_chart(fig, use_container_width=True)

        # AI Summary
        st.subheader(" AI-Generated Repo Summary")
        text_to_summarize = (r.get("description") or "") + " " + \
                            f"This repository has {r['stargazers_count']} stars, {r['forks_count']} forks, and {r['open_issues_count']} open issues."

        if text_to_summarize.strip():
            input_len = len(text_to_summarize.split())
            max_len = min(60, input_len + 10)  # dynamic length
            try:
                summary = summarizer(
                    text_to_summarize,
                    max_length=max_len,
                    min_length=10,
                    do_sample=False
                )
                st.info(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Summarization failed: {e}")
