import os
import streamlit as st
from langchain_cohere.chat_models import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from bs4 import BeautifulSoup, Comment
import requests
from dotenv import load_dotenv

load_dotenv()

# Initialize Summarization Model
summarization_model = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"), model="command-r-plus", temperature=0.5)

embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0")

mongo_client = MongoClient(host=os.getenv("ATLAS_CONNECTION_STRING"))

# Define MongoDB Database and Collection
webpages_database = mongo_client["webpages"]
content_collection = webpages_database["content"]

vectorstore = MongoDBAtlasVectorSearch(collection=content_collection, embedding=embeddings_model, index_name="content_index")

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_useful_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        elements = soup.find_all(['p', 'div', 'article', 'section'])

        visible_texts = filter(tag_visible, elements)
        content_parts = [element.get_text(separator=' ').strip() for element in visible_texts if element.get_text(strip=True)]

        main_content = ' '.join(content_parts)
        main_content = ' '.join(main_content.split())

        return main_content.strip()

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def summarize_content(content):
    prompt_template = PromptTemplate.from_template(template="Summarize the following content:\n\n{content}\n\nSummary:")
    response = summarization_model.invoke(prompt_template.format(content=content))
    return response.content

def store_content_in_db(url, content, summary):
    ingestion_docs = []
    total_documents = content_collection.count_documents({})
    id = total_documents + 1
    if content:
        ingestion_docs.append(Document(page_content=content, metadata={'id': id, 'url': url, 'summary': summary}))
        # id += 1
    vectorstore.add_documents(ingestion_docs)

def summarize_webpage(url):
    raw_content = extract_useful_content(url)
    if not raw_content:
        return None

    summary = summarize_content(raw_content)
    store_content_in_db(url, raw_content, summary)
    return summary

def get_answer(query):
    qa_prompt_template = PromptTemplate.from_template("Answer the question based on the provided sources only: {context}\n\nQuestion: {query}")
    str_parser = StrOutputParser()
    search_result = vectorstore.similarity_search(query=query, k=2)
    context = ''
    doc_id = 1
    for doc in search_result:
        context += f'\n\nSource {doc_id}:\n' + doc.metadata['summary']
        doc_id += 1
    qa_chain = qa_prompt_template | summarization_model | str_parser
    response = qa_chain.invoke({'context': context, 'query': query})
    return response

# Streamlit Interface
st.title("Webpage Summarizer and Q&A System")

# Use session state to store the summary and answer
if "summary" not in st.session_state:
    st.session_state.summary = None
if "answer" not in st.session_state:
    st.session_state.answer = None

# Function to clear the summary
def clear_summary():
    st.session_state.summary = None

# Function to clear the answer
def clear_answer():
    st.session_state.answer = None

# Input URL and summarize
url = st.text_input("Enter URL of the webpage to summarize:", "", key="url", on_change=clear_summary)
url_submit_button = st.button("Summarize Webpage")

if url_submit_button and url:
    with st.spinner("Summarizing webpage..."):
        summary = summarize_webpage(url)
        if summary:
            st.session_state.summary = summary
        else:
            st.write("Failed to summarize the webpage.")
elif url_submit_button:
    st.warning("Please enter a URL.")

# Display summary
if st.session_state.summary:
    st.subheader("Summary:")
    st.write(st.session_state.summary)

# Input query and get answer
query = st.text_input("Enter your question:", key="query", on_change=clear_answer)
query_submit_button = st.button("Get Answer")

if query_submit_button and query:
    with st.spinner("Getting answer..."):
        answer = get_answer(query)
        st.session_state.answer = answer
elif query_submit_button:
    st.warning("Please enter a question.")

# Display answer
if st.session_state.answer:
    st.subheader("Answer:")
    st.write(st.session_state.answer)