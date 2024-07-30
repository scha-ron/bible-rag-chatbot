import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import DB_PATH, EMBEDDING_MODEL
from preprocess import BIBLE_VERSION
import re
import os
from typing import List, Tuple, Dict, NamedTuple
import numpy as np
from sentence_transformers import SentenceTransformer
# import requests
# from functools import lru_cache
#from langchain_ollama import OllamaEmbeddings


st.title(f"Bible RAG Chatbot en - ({BIBLE_VERSION.upper()})")

# Define a named tuple for examples
class Example(NamedTuple):
    question: str
    answer: str

# Load sentence transformer model for semantic similarity
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# Expanded set of examples 
EXAMPLES = [
    Example("Can you tell me what the Bible says about love?",
            "The Bible speaks extensively about love. One of the most famous passages is found in 1 Corinthians 13:4-7, which beautifully describes the characteristics of love: \"Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up, Doth not behave itself unseemly, seeketh not her own, is not easily provoked, thinketh no evil; Rejoiceth not in iniquity, but rejoiceth in the truth; Beareth all things, believeth all things, hopeth all things, endureth all things.\""),
    Example("I'm curious about Moses. Who was he exactly?",
            "Moses was a key figure in the Bible - a Hebrew prophet, leader, and lawgiver. He was born in Egypt when Pharaoh had ordered all male Hebrew infants to be killed. His mother placed him in a basket among the reeds by the riverbank, where Pharaoh's daughter found and adopted him. God appeared to Moses in a burning bush, calling him to lead the Israelites out of Egypt. He received the Ten Commandments from God on Mount Sinai. Moses led the Israelites out of Egypt, performed miracles, and guided them through the wilderness for forty years. He died at the age of 120 on Mount Nebo, after God showed him the promised land."),
    Example("Why is the Resurrection so important in Christianity?",
            "The Resurrection of Jesus Christ is a cornerstone of Christian faith. It refers to Jesus rising from the dead on the third day after his crucifixion. The Apostle Paul emphasizes its importance in 1 Corinthians 15:14, stating: \"And if Christ be not risen, then is our preaching vain, and your faith is also vain.\" The Resurrection symbolizes victory over death and sin, and is central to the promise of eternal life for believers."),
    Example("How does the Bible teach us about forgiveness?",
            "Forgiveness is a key theme in the Bible. Jesus taught extensively on this subject, emphasizing its importance in the Lord's Prayer and in parables. In Matthew 6:14-15, He says: \"For if ye forgive men their trespasses, your heavenly Father will also forgive you: But if ye forgive not men their trespasses, neither will your Father forgive your trespasses.\" The Bible encourages believers to forgive others as they have been forgiven by God."),
    Example("I've heard about the Great Commission. What exactly is it?",
            "The Great Commission is a directive given by Jesus to his disciples, found in Matthew 28:19-20: \"Go ye therefore, and teach all nations, baptizing them in the name of the Father, and of the Son, and of the Holy Ghost: Teaching them to observe all things whatsoever I have commanded you: and, lo, I am with you always, even unto the end of the world. Amen.\" It is considered a fundamental mission for Christians to spread the Gospel and make disciples throughout the world."),
    Example("Could you share what Habakkuk 3:1-5 says?",
            "Certainly. Habakkuk 3:1-5 in the King James Version reads: '1 A prayer of Habakkuk the prophet upon Shigionoth. 2 O Lord, I have heard thy speech, and was afraid: O Lord, revive thy work in the midst of the years, in the midst of the years make known; in wrath remember mercy. 3 God came from Teman, and the Holy One from mount Paran. Selah. His glory covered the heavens, and the earth was full of his praise. 4 And his brightness was as the light; he had horns coming out of his hand: and there was the hiding of his power. 5 Before him went the pestilence, and burning coals went forth at his feet.'")
]

# Precompute embeddings for example questions
example_embeddings = sentence_model.encode([ex.question for ex in EXAMPLES])
def get_relevant_examples(question: str, num_examples: int = 2) -> List[Dict]:
    question_embedding = sentence_model.encode([question])[0]
    similarities = np.dot(example_embeddings, question_embedding) / (
        np.linalg.norm(example_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = similarities.argsort()[-num_examples:][::-1]
    return [EXAMPLES[i] for i in top_indices]

@st.cache_resource
def load_bible_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.stop()

with st.spinner("Loading Bible knowledge..."):
    vectorstore = load_bible_vectorstore()
    retriever = vectorstore.as_retriever()

# Ollama LLM config
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
main_llm = ChatOllama(model='llama3', base_url=ollama_host)
verification_llm = ChatOllama(model='gemma2', base_url=ollama_host)  

# Uncomment if running directly from the repo after cloning
# main_llm = ChatOllama(model='llama3')

# # Verification LLM
# verification_llm = ChatOllama(model='gemma2')

def is_full_chapter_request(question: str) -> bool:
    return "entire" in question.lower() or "full" in question.lower() or "whole" in question.lower()

def extract_book_and_chapter(question: str) -> Tuple[str, int]:
    pattern = r'(\w+)\s+(\d+)'
    match = re.search(pattern, question)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def is_greeting(text):
    greeting_patterns = r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b'
    return bool(re.search(greeting_patterns, text.lower()))

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a deeply insightful Bible AI-assistant, intimately familiar with the Bible.
     **You are based exclusively EXCLUSIVELY on the King James Version (KJV) of the Bible.**
     Your purpose is to guide individuals to explore and embrace the wisdom and teachings within God's Word.
     Use the retrieved context to answer questions and give more detailed information.
     **Check the context carefully before answering to ascertain relevance, accuracy, and correctness. If a question cannot be answered solely with the retrieved context, you can supplement with your knowledge of the KJV, BUT **clearly** indicate when you're doing so.**
      **Only use this internal knowledge for Bible-specific information. It MUST be from the KJV.** 
     VERY IMPORTANT: If you do not know the answer to a question, simply say so POLITELY and FIRMLY, DO NOT make up/conjure your own answer.
     If the question is not about the Bible, politely decline to answer.
     Provide a clear and concise answer to the user's question without including references or citations in your response.
     Be **precise** in your explanations and ensure that any biblical quotes or paraphrases **accurately** reflect the specific verses they come from.
     VERY IMPORTANT: When quoting from the scriptures, quote it EXACTLY AS IT APPEARS IN THE KING JAMES VERSION (KJV)! **Do not** modernize or alter the original language in **any** way. 
     The system will handle verifying and adding references separately.
     If asked for a full chapter, provide the entire chapter text **with verse numbers**.
     
     Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def secondary_verification(response, citations, question):
    is_full_chapter_req = is_full_chapter_request(question)
    book, chapter = extract_book_and_chapter(question)
    
    verification_prompt = f"""
    You are a Bible expert specializing in the King James Version (KJV). Your task is to verify and correct the following response to ensure it matches the KJV exactly, including all verse numbers if applicable.

    Original question: {question}
    Response to verify:
    {response}

    Please **verify and correct the response** to ensure it accurately reflects the KJV. If it's a full chapter request, **make sure all verses are included and numbered correctly as they are in the KJV**.

    Format your response as follows:
    Verified Text:
    [Corrected text, with verse numbers for full chapters]

    Citations:
    [Book Chapter:Verse range]
    """
    
    result = verification_llm.invoke(verification_prompt)
    verified_text = result.content.split("Verified Text:", 1)[-1].split("Citations:", 1)[0].strip()
    
    if is_full_chapter_req and book and chapter:
        verified_citations = [f"{book} {chapter}:1-{len(verified_text.split())}"]
    else:
        verified_citations = citations
    
    return verified_text, verified_citations

def rag_chain(question: str, chat_history: list):
    if is_greeting(question):
        return "Hello! How can I assist you with your Bible-related questions today?"

    relevant_docs = retriever.invoke(question)
    initial_citations = [f"{doc.metadata['book']} {doc.metadata['chapter']}:{doc.metadata['verse']}" for doc in relevant_docs]

    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | main_llm
        | StrOutputParser()
    )
    
    initial_response = chain.invoke({"question": question, "chat_history": chat_history})
    
    final_response, final_citations = secondary_verification(initial_response, initial_citations, question)
    
    if final_citations:
        return f"{final_response.strip()}\n\nNotes and References:\n" + "\n".join(final_citations)
    return final_response.strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about the Bible:"):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            try:
                response = rag_chain(user_input, chat_history)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                fallback_response = main_llm.invoke(f"Provide a brief answer about {user_input} from the King James Bible perspective, without quoting specific verses.")
                st.markdown(f"I apologize, but I encountered an error while processing your question. Here's a general response:\n\n{fallback_response}")
                response = f"Error occurred. Fallback response: {fallback_response}"

    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info("This chatbot uses RAG to answer questions about the Bible.")