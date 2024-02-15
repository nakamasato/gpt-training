# Output Parser

## Prompt
1. Prepare `ResponseSchema`
    ```py
    gift_schema = ResponseSchema(name="gift",
                                description="Was the item purchased\
                                as a gift for someone else? \
                                Answer True if yes,\
                                False if not or unknown.")
    ```
1. Craete OutputParser

    ```py
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    ```
1. Generate format instructions from the OutputParser
    ```py
    format_instructions = output_parser.get_format_instructions()
    ```
1. Chat
1. Parse
    ```
    output_dict = output_parser.parse(response.content)
    ```
## Run

```
poetry run python src/langchain/output_parser.py
```

```
{'gift': True, 'delivery_days': '2', 'price_value': ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]}
```
