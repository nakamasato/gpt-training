# Testing

## Installation

```
poetry add pytest pytest-cov pytest-env --group dev
```

## Run

```
poetry run pytest
```

## Fake

```py
from langchain.llms.fake import FakeListLLM
```
