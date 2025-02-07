import os
import re
import openai
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain.prompts import ChatPromptTemplate

## API Configuration
openai.api_key = "AaRMQBv5rxJifjrk4IOiq1jggKs34xupXhjpkuCotP2YzADMHQ7TJQQJ99BAACHYHv6XJ3w3AAAAACOGjMeX"
openai.api_type = "azure"
openai.api_base = "https://germa-m5jyh5dj-eastus2.openai.azure.com/"
openai.api_version = "2024-05-01-preview"
engine = "gpt-4"

## Function to call Azure LLM
def azure_llm(temperature, max_tokens, prompt):

    response = openai.ChatCompletion.create(
            engine = engine,
            messages = prompt,
            temperature = temperature,
            max_tokens  = None,
            top_p=0.2,
            frequency_penalty = 0,
            presence_penalty = 0
            )

    text = response.choices[0].message.content
    return text

def final_response (review, lang):
    temperature = 0.1
    max_tokens = None
    # review = translator(review, lang)

    prompts = f""" Analyze the sentiment of the given text and return a numerical value based on the sentiment type:
                - Neutral: 0
                - Anger: 1
                - Surprise: 2
                - Happy: 3
                - Sad: 4

            Text: {review}
            Sentiment (numerical value)

            if result == "0":
                ans = "Sentiment of text is Neutral 😑"
            elif result == "1":
                ans = "Sentiment of text is Angery 😡"
            elif result == "2":
                ans = "Sentiment of text is Surprised 😱"
            elif result == "3":
                ans = "Sentiment of text is Happy 😊"
            else:
                ans = "Sentiment of text is Sad 😢"

            Please return the ans text in {lang} language
            """

    prompt= ChatPromptTemplate.from_template(prompts).format_messages(review = review)
    message_text=[{"role": "system", "content": prompt[0].content}]
    resp = azure_llm(temperature, max_tokens, message_text)
    return resp


## Analysis of count of positive and negative reviews from the data
def plot_data1(data):
    df = data.value_counts('label')
    df.index = df.index.map(lambda x: 'positive' if x == 1 else 'negative')
    return df


## Analysis based on rating
def plot_data2(data):
    df = data.value_counts('rating')
    return df


## Function to clean the text
def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove single characters
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)
    # Lowercase the text
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip()
    return text

#Sidebar for operation selection
operation = st.sidebar.selectbox("Select Operation", ["Text Analyze", "File Analyze"])

#Text Analyze Section
if operation == "Text Analyze":
    lang = st.sidebar.selectbox("Select Language", ["English", "Spanish", "German", "Roman"])
    st.title("Text Sentiment Analysis")
    st.markdown('''This application is all about sentiment analysis of inputed texts.
            We can analyse reviews of the customer using this application.''')
    with st.form("text_analyze_form"):
        text_input = st.text_area("Enter text to analyze")
        model_option = st.selectbox("Select model", ["GPT-4.0"])
        analyze_button = st.form_submit_button("Analyze")

    if analyze_button:
        if text_input:
            print(lang)
            result = final_response(text_input, lang)
            st.write(result)

        else:
            st.warning("Please enter text to analyze.")

## File Analyze Section
elif operation == "File Analyze": 
    st.sidebar.title("Sentiment Analysis Of Customer Reviews")
    st.markdown("We can analyse reviews of the customer using this application")
    
    # File Upload
    uploaded_file =  st.sidebar.file_uploader("Choose a file")

    if uploaded_file:
        save_folder = "uploads"
        os.makedirs(save_folder, exist_ok=True)

        file_path = os.path.join(save_folder, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ans = f"File saved to {file_path}"

    # File Selection for Analysis

    st.subheader("Select the file which you want to analyse")
    files = os.listdir("uploads") if os.path.exists("uploads") else []
    selected_file = st.selectbox("Select a file", files)

    if st.button("Analyse"):
        if selected_file:
            file_path = os.path.join("uploads", selected_file)
            data = pd.read_csv(file_path, sep='\t')
            df = data[['rating', 'verified_reviews', "feedback"]]
            df.columns = ['rating', 'review', 'label']
            st.write(df.head(20))

            st.subheader("Analysis of count of positive and negative reviews from the data")
            sentiment = plot_data1(df)
            sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Reviews': sentiment.values})
            fig =  px.bar(sentiment, x='Sentiment', y='Reviews', color='Reviews', height=500)
            st.plotly_chart(fig)

            fig = px.pie(sentiment, values = 'Reviews', names = "Sentiment")
            st.plotly_chart(fig)

            st.subheader("Analysis of sentiments on basis of ratings from the data")
            sentiment = plot_data2(df)
             
            sentiment = pd.DataFrame(
                {'Sentiment': sentiment.index, 'Reviews': sentiment.values}
            )
            fig =  px.bar(sentiment, x='Sentiment', y='Reviews', color='Reviews', height=500)
            st.plotly_chart(fig)
            
        else:
            st.warning("No file selected")