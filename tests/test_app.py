import requests

# 1. Sanity check GitHub API call
def test_github_api_repo_metadata():
    url = "https://api.github.com/repos/streamlit/streamlit"
    r = requests.get(url)
    assert r.status_code == 200
    data = r.json()
    assert "stargazers_count" in data
    assert "forks_count" in data
    assert "open_issues_count" in data

# 2. HuggingFace summarizer smoke test
def test_huggingface_summarizer():
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = "Streamlit makes it easy to build data apps in Python. It is popular among data scientists."
    summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
    assert isinstance(summary, list)
    assert "summary_text" in summary[0]
    assert len(summary[0]["summary_text"]) > 0
