from streamlit.testing.v1 import AppTest


def test_app_user_summary_home():
    at = AppTest.from_file("src/projects/apps/streamlit/app.py")
    at.run(timeout=10)
    assert not at.exception
    assert at.title[0].value == "Langchain Sample App"
