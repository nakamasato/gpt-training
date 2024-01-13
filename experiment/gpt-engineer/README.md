# gpt-engineer

## Environment Variables

1. `OPENAI_API_KEY`

## Setup

```
brew install python-tk
```

```
poetry add gpt-engineer
```

## Usage

1. Write `main_prompt` in project dir
    Example:
    ```
    Please write a golang script to apply sql based atlas migration files (database migrations).
    ```
1. Run
    ```
    poetry run gpt-engineer <projec path>
    # example: poetry run gpt-engineer experiment/gpt-engineer/sample
    ```

> [!NOTE]
> If you encounter `ModuleNotFoundError: No module named '_tkinter'`, you might need to reinstall your Python `pyenv uninstall xxx && pyenv install xxx` after installing tkinter.

## sample

Didn't work well...

## Ref

1. https://gpt-engineer.readthedocs.io/en/latest/index.html
1. https://qiita.com/M0304/items/0621c45a14db10931097

