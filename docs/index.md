# GPT Training

## Project layout

```bash
experiment/ # experiment for new tools
├── gpt-engineer
│   ├── README.md
src/ # src for runnable scripts and apps
├── scripts/ # one-time scripts
├── libs/ # common libs
├── apps/ # runnable apps
docs/ # md files for mkdocs
├── index.md
README.md
mkdocs.yml
```

## Python

### Environment Variables

`.env`

```
pip install poetry-dotenv-plugin
```

### Testing

```
poetry run pytest
```
