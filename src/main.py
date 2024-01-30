# TODO: make application startup 

import streamlit as st
import numpy as np
import random
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


load_dotenv()

print("Loading vectorstore...")
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='../vectorstore/crescent')
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

template = """Answer the question based on the following context.

{context}

Question: {question}

If there is no information in the context, think rationally and provide an answer based on your own knowledge.
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_answer = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_sources = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_answer)

def format_response(resp):
    src = resp["context"][0].metadata["source"].split('\\')[-1]
    span_start = resp["context"][0].metadata["start_index"]
    span_end = span_start + len(resp["context"][0].page_content)
    
    response = f"""{resp['answer']}
  
Source: [{src}](file:///C:\d\Work\GitHub\chat-assistant\src\{resp["context"][0].metadata["source"]})
Page: {resp['context'][0].metadata['page']}, span: {span_start}--{span_end}
"""
    return response

def get_response(prompt):
    resp = rag_chain_with_sources.invoke(prompt)

    return format_response(resp)

print("Loading complete.")

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        assistant_response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        st.write(assistant_response)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if prompt := st.chat_input("Message Chatbot..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = get_response(prompt)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split('\n'):
            for wrd in chunk.split(' '):
                    full_response += wrd + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.write(full_response + "▌")
            full_response += "  \n"

        message_placeholder.write(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    