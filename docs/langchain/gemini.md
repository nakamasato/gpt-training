# Gemini

[ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)

## Pricing

60 QPM (queries per minute)

Details: https://ai.google.dev/pricing

## Set up API Key

1. Set project id
    ```
    PROJECT=xxx
    KEY_ID=gemini-api-key
    ```

1. Enable service `generativelanguage.googleapis.com`

    ```
    gcloud services list --available --project $PROJECT --filter 'generativelanguage'
    NAME                               TITLE
    generativelanguage.googleapis.com  Generative Language API
    ```

    ```
    gcloud services enable generativelanguage.googleapis.com --project $PROJECT
    ```

1. Set alias ([ref](https://cloud.google.com/service-usage/docs/set-up-development-environment))

    ```
    alias gcurl='curl -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json"'
    ```

1. Create API key with API target restriction.

    ```
    DISPLAY_NAME="Gemini API Key"
    gcloud services api-keys create --display-name "$DISPLAY_NAME" --project $PROJECT --api-target=service=generativelanguage.googleapis.com
    ```

1. Check API Key
    ```
    gcloud services api-keys list --project $PROJECT --format json | jq ".[] | select(.displayName == \"$DISPLAY_NAME\")"
    {
      "createTime": "2024-02-23T12:58:04.334290Z",
      "displayName": "Gemini API Key",
      "etag": "W/\"dMMHgHoJFaTkh6z89v139Q==\"",
      "name": "projects/xxxxx/locations/global/keys/xxxxxxx",
      "restrictions": {
        "apiTargets": [
          {
            "service": "generativelanguage.googleapis.com"
          }
        ]
      },
      "uid": "xxxxxxxxxxxxxxx",
      "updateTime": "2024-02-23T12:58:04.358444Z"
    }
    ```

1. Get Key string

    ```
    gcloud services api-keys get-key-string projects/xxxxxxx/locations/global/keys/xxxxxxxxxxxxxxxx --project $PROJECT
    keyString: XXXXXXXXXXXXXXXXXXXXX
    ```

    Set this value in `.env`

## Ref

1. https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
1. https://cloud.google.com/api-keys/docs/add-restrictions-api-keys
1. https://ai.google.dev/pricing
1. https://makersuite.google.com/app/
1. [[langchain] ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
1. https://qiita.com/nakamasato/items/b8d76da6751df8d7bddc
