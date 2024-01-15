import openai
import os
import streamlit as st

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_article(title: str, summary: str) -> str:
    # Template for the prompt
    prompt_template = "
    # {title}
    ## {summary}
    Please write an article on the above topic.
    "
    # Filling the template
    prompt = prompt_template.format(title=title, summary=summary)

    # Use GPT-3 to generate a response
    response = openai.Completion.create(
       engine="text-davinci-001",
       prompt=prompt,
       temperature=0.5,
       max_tokens=1000
    )

    # Extract the generated article text
    article = response.choices[0].text.strip()

    # Return the generated article
    return article

def main():
    st.title("GPT-3 Article generator")
    st.header("Input the title and summary to generate article")
    # Text input for the title
    title = st.text_input('Title')
    # Text area for the summary
    summary = st.text_area('Summary', height=200)
    # Button to generate the article
    if st.button('Generate Article'):
        article = generate_article(title, summary)
        st.text_area('Generated Article', article, height=400)

if __name__ == "__main__":
    main()
