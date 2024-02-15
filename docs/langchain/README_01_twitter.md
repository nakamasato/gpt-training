# [LangChain: Twitter algorithm code](https://github.com/langchain-ai/langchain/blob/master/cookbook/twitter-the-algorithm-analysis-deeplake.ipynb)


embeddings -> DeepLake (Vector Store) -> Question Answering

## Components

1. Splitter
1. [Embedding](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
1. [VectorStore](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
    1. [DeepLake](https://github.com/activeloopai/deeplake)
    1. [Chroma](https://github.com/chroma-core/chroma)
    1. [Qdrant](https://github.com/qdrant/qdrant)
1. [Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/)

## Run

```
poetry example --example langchain-twitter-algorithm
```

<details>

```
-> **Question**: What does favCountParams do?

**Answer**: It is not clear from the given context what exactly `favCountParams` does. It seems to be an optional parameter of `ThriftLinearFeatureRankingParams` in a larger codebase that includes many other parameters and modules. Without more context or documentation, it is impossible to say what this parameter specifically does.

-> **Question**: is it Likes + Bookmarks, or not clear from the code?

**Answer**: Based on the provided context, `favCountParams` is one of the optional `ThriftLinearFeatureRankingParams` in the codebase. However, without further context, it is unclear what its specific function is within the larger codebase.

-> **Question**: What are the major negative modifiers that lower your linear ranking parameters?

**Answer**: I'm sorry, but I can't provide a specific answer to your question without more context. The code you provided includes multiple references to different linear ranking parameters and modifiers, so I would need to know which specific parameters and modifiers you are referring to. Could you please provide more specific context or clarify your question?

-> **Question**: How do you get assigned to SimClusters?

**Answer**: Unfortunately, the given context does not provide information about how one gets assigned to SimClusters. The paper describes SimClusters as a general-purpose representation layer based on overlapping communities into which users as well as heterogeneous content can be captured as sparse, interpretable vectors to support a multitude of recommendation tasks. However, it does not provide information on the assignment process.

Retrying langchain.chat_models.openai.ChatOpenAI.completion_with_retry.<locals>._completion_with_retry in 1.0 seconds as it raised RateLimitError: That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at help.openai.com if the error persists. (Please include the request ID 4b9f981cfd37f488b2dcfd71a14a5792 in your message.).
-> **Question**: What is needed to migrate from one SimClusters to another SimClusters?

**Answer**: Can you explain the difference between SimClusters Scalding Jobs and SimClusters GCP Jobs?

-> **Question**: How much do I get boosted within my cluster?

**Answer**: I'm sorry, but I couldn't find enough information to answer your question about the amount of boost that one receives within their SimCluster. Could you please provide me with more context or clarify your question?

-> **Question**: How does Heavy ranker work. what are it’s main inputs?

**Answer**: The Heavy Ranker is part of the Follow Recommendations Service (FRS), which is a service that generates a list of candidate accounts for a user to follow. The Heavy Ranker employs both Machine Learning (ML) and heuristic rule-based candidate ranking. The main inputs to the Heavy Ranker are the scoring algorithm, source embedding ID, candidate embedding type, minimum score, and candidates. The candidates are pairs of <user, candidate> and a DataRecord is constructed for each pair. The ML features are fetched beforehand (i.e., feature hydration) and then sent to a separate ML prediction service, which houses the ML model trained offline. The ML prediction service returns a prediction score, representing the probability that a user will follow and engage with the candidate. This score is a weighted sum of p(follow|recommendation) and p(positive engagement|follow), and FRS uses this score to rank the candidates.

-> **Question**: How can one influence Heavy ranker?

**Answer**: Unfortunately, the context provided does not mention any specific methods to influence the Heavy Ranker in the Follow Recommendations Service. The Heavy Ranker employs Machine Learning and heuristic rule-based candidate ranking to rank the candidates. The Machine Learning model is trained offline, and its prediction score represents the probability that a user will follow and engage with the candidate. If there are any specific methods to influence the Heavy Ranker, they are not mentioned in the provided context.

-> **Question**: why threads and long tweets do so well on the platform?

**Answer**: I'm sorry, based on the context provided, I cannot find any information about why threads and long tweets perform well on the platform. The context only provides information about the high-level architecture and main components of the Tweet Search System (Earlybird) and how it works to retrieve in-network tweets.

-> **Question**: Are thread and long tweet creators building a following that reacts to only threads?

**Answer**: I'm sorry, but I cannot determine an answer to your question based on the given context. The code snippets provided are related to conversation controls and following features, and don't provide any information about creators who post long threads on the platform and their following interactions.

-> **Question**: Do you need to follow different strategies to get most followers vs to get most likes and bookmarks per tweet?

**Answer**: Unfortunately, the given context does not provide information on the different strategies to get the most followers on the platform versus getting the most likes and bookmarks per tweet. The provided context mainly discusses the algorithms and services used by Twitter to rank users and tweets, and how these services are used to enhance the user experience.

-> **Question**: Content meta data and how it impacts virality (e.g. ALT in images).

**Answer**: I'm sorry, but the given context does not provide any information on how content metadata, such as ALT in images, affects virality on the Twitter platform. The context mainly discusses the architecture of the Tweet Search System (Earlybird) and various rules used by the visibility library for tweet and user ranking.

-> **Question**: What are some unexpected fingerprints for spam factors?

**Answer**: I'm sorry, but I don't see any context or information related to "unexpected fingerprints for spam factors". Can you please provide more context or clarify your question?

-> **Question**: Is there any difference between company verified checkmarks and blue verified individual checkmarks?

**Answer**: Based on the code provided, it seems that the system is checking if a Twitter user is "blue verified", which means that they have been personally verified by Twitter as an individual. Company verified checkmarks, on the other hand, indicate that a business or organization's Twitter account has been verified by Twitter. So, the difference is that blue verified checkmarks are for individuals, while company verified checkmarks are for businesses or organizations.
```

</details>

## Ref

- https://python.langchain.com/en/latest/reference/modules/chat_models.html#langchain.chat_models.ChatOpenAI
- https://github.com/langchain-ai/langchain/blob/master/cookbook/twitter-the-algorithm-analysis-deeplake.ipynb
