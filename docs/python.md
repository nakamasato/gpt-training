# Python

## Environment Variables

`.env`

```
pip install poetry-dotenv-plugin
```

## Testing

### Installation

```
poetry add pytest pytest-cov pytest-env --group dev
```

### Run

```
poetry run pytest
```

### patch

Source code:

```py
from langchain_community.utilities import GoogleSearchAPIWrapper
google = GoogleSearchAPIWrapper()
```

Test:

```py
@patch("src.libs.tools.google")
def test_xxx(mock_google):
    google_mock.results.return_value = [
        {
            "title": "内閣制度と歴代内閣",
            "link": "https://www.kantei.go.jp/jp/rekidai/ichiran.html",
            "snippet": "(外務大臣 内田康哉が内閣総理大臣臨時兼任）. 22, （第2次） 山本權兵衞, 大正12.9.2 －大正13.1.7, 128, 70歳, 嘉永5.10.15, 昭和8.12.8 (81歳), 鹿児島県, 549. 23(13)\xa0...",
        },
    ]
```

### FakeListLLM

[langchain/testing.md](langchain/testing.md)
