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

1. Enable service

    ```
    gcloud services list --available --project $PROJECT --filter 'aiplatform'
    NAME                                                TITLE
    aiplatform.googleapis.com                           Vertex AI API
    contactcenteraiplatform.googleapis.com              Contact Center AI Platform API
    ml.googleapis.com                                   AI Platform Training & Prediction API
    scai-platform-2-scaidata.cloudpartnerservices.goog  ScaiData ScaiPlatform - Free Real-Time Business Intelligence & Analytics
    ```

    ```
    gcloud services enable aiplatform.googleapis.com --project $PROJECT
    ```

1. Set alias ([ref](https://cloud.google.com/service-usage/docs/set-up-development-environment))

    ```
    alias gcurl='curl -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json"'
    ```

1. Create API key

    ```
    gcurl https://apikeys.googleapis.com/v2/projects/$PROJECT/locations/global/keys?keyId=$KEY_ID -X POST -d '{"displayName" : "Gemini API key"}'
    ```

1. Add restriction
    ```
    gcurl https://apikeys.googleapis.com/v2/projects/$PROJECT/locations/global/keys/$KEY_ID\?updateMask\=restrictions \
      --request PATCH \
      --data '{
        "restrictions" : {
          "browserKeyRestrictions": {
            "allowedReferrers": "aiplatform.googleapis.com"
          }
        }
      }'
    ```

1. Check API Key
    ```
    gcurl https://apikeys.googleapis.com/v2/projects/$PROJECT/locations/global/keys/$KEY_ID
    {
      "name": "projects/742402882542/locations/global/keys/gemini-api-key",
      "displayName": "Gemini API key",
      "createTime": "2024-02-10T07:42:35.909516Z",
      "uid": "c0169fc9-ab4f-4e8d-b8b9-438dcf7f82ef",
      "updateTime": "2024-02-10T08:08:22.817143Z",
      "restrictions": {
        "browserKeyRestrictions": {
          "allowedReferrers": [
            "aiplatform.googleapis.com"
          ]
        }
      },
      "etag": "W/\"zMOkAKcUizsvNs3NdtX+sA==\""
    }
    ```

1. Get Key string

    ```
    gcurl https://apikeys.googleapis.com/v2/projects/$PROJECT/locations/global/keys/$KEY_ID/keyString
    ```

    Set this value in `.env`

## Ref

1. https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
1. https://cloud.google.com/api-keys/docs/add-restrictions-api-keys
1. https://ai.google.dev/pricing
1. https://makersuite.google.com/app/
1. [[langchain] ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
