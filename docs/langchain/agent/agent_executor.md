# AgentExecutor

`AgentExecutor` executes an `Agent` with a set of `Tools` and a `CallbackManager`.

## How to Use

```py
from langchain.agent import AgentExecutor

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    callback_manager=None, # default: None (explicitly write None)
    verbose=True,
)
```

## Implementation

1. An agent is called by an [AgentExecutor](https://github.com/langchain-ai/langchain/blob/0d888a65cb1b89540c02db3a743db03a91cf4fa5/libs/langchain/langchain/agents/agent.py#L915) (`AgentExcutor` -> `Chain` -> `RunnableSerializable`) which is a `Chain` that takes an `agent` and `tools` as inputs.

    ```py
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=False)
    ```

    1. `memory` is optional.
    1. `verbose` is `True` by default.
    1. `handle_parsing_errors` is `False` by default.
1. Invoke the agent with `agent_executor.invoke({"input": "Who is Japan's prime minister?")`.
    1. [invoke](https://github.com/langchain-ai/langchain/blob/0d888a65cb1b89540c02db3a743db03a91cf4fa5/libs/langchain/langchain/chains/base.py#L119) is defined in [Chain](https://github.com/langchain-ai/langchain/blob/0d888a65cb1b89540c02db3a743db03a91cf4fa5/libs/langchain/langchain/chains/base.py#L43).
    1. `invoke` calls an abstractmethod [_call](https://github.com/langchain-ai/langchain/blob/0d888a65cb1b89540c02db3a743db03a91cf4fa5/libs/langchain/langchain/chains/base.py#L287).

        ```py
        @abstractmethod
        def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Dict[str, Any]:
            """Execute the chain.

            This is a private method that is not user-facing. It is only called within
                `Chain.__call__`, which is the user-facing wrapper method that handles
                callbacks configuration and some input/output processing.

            Args:
                inputs: A dict of named inputs to the chain. Assumed to contain all inputs
                    specified in `Chain.input_keys`, including any inputs added by memory.
                run_manager: The callbacks manager that contains the callback handlers for
                    this run of the chain.

            Returns:
                A dict of named outputs. Should contain all outputs specified in
                    `Chain.output_keys`.
            """
        ```

1. The actual implementation of `_call` for `AgentExecutor` is [here](https://github.com/langchain-ai/langchain/blob/0d888a65cb1b89540c02db3a743db03a91cf4fa5/libs/langchain/langchain/agents/agent.py#L1413)

    ```py
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)
    ```

    1. There's a `while` loop, which iterates the following logic if `_should_continue` returns true.
    1. Call `_take_next_step` and get `next_step_output`, inside which the agent calls `plan` and the tool is executed.
        1. Plan: Decide which tool to use
        1. Execute: Execute the tool
    1. If the `next_step_output` is an `AgentFinish`, it returns the output.
    1. Otherwise, it extends the `intermediate_steps` and checks if the tool should return directly.
    1. It increments the iterations and time elapsed.
    1. If the loop ends, it returns the output.
