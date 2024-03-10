# [Caching](https://python.langchain.com/docs/modules/model_io/models/llms/llm_caching)

```py
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

callbackを使っているときには、[BaseCallbackHandler.on_llm_end](https://api.python.langchain.com/en/latest/callbacks/langchain.callbacks.base.BaseCallbackHandler.html#langchain.callbacks.base.BaseCallbackHandler.on_llm_end)で値を取得することができる。

streamlitでStreamingかつCacheを使っている場合 [StreamlitCallbackHandler](https://python.langchain.com/docs/integrations/callbacks/streamlit)
