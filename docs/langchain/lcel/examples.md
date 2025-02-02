# LCEL examples

## RunnableBranch

You can use `RunnableBranch` to create a conditional chain.

- if `message` is not empty, then run `message_summary_chain`
- if `message` is empty, then return `"no message found"`

```python
def example_runnable_branch():
    # branch
    extract_message_chain = RunnableLambda(lambda x: {"message": ["message"] * randint(0, 2)})  # randomize message
    message_summary_chain = RunnableLambda(lambda x: f"this is the summary of {len(x['message'])} messages")  # emulate chain with LLM

    # chain
    chain_with_branch = extract_message_chain | RunnableBranch(
        (lambda x: bool(x["message"]), message_summary_chain),
        lambda x: "no message found",  # fixed result when empty
    )
    print(chain_with_branch.batch([{}, {}, {}]))
```

## More examples

```py
--8<-- "src/langchain/runnable_custom.py"
```
