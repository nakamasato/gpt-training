# Gemini API

## Setup

```
PROJECT=your-project-id
gcloud services enable generativelanguage.googleapis.com --project $PROJECT
gcloud services api-keys create --display-name "Gemini API Key" --api-target=service=generativelanguage.googleapis.com --project $PROJECT
```

## Run

```
poetry run python experiment/gemini/main.py
こんにちは！ご機嫌はいかがですか？
```

## References

1. https://cloud.google.com/natural-language/docs/setup
1. https://zenn.dev/peishim/articles/2e2e8408888f59
