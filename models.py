import streamlit as st
import os
from sentence_transformers import CrossEncoder
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

hg_endpoint_url = "https://vbjn6efeqofjonee.us-east-1.aws.endpoints.huggingface.cloud"

generation_config = {"temperature": 0.1, "top_p": 0.9, "max_tokens": 512}


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="multi-qa-mpnet-base-dot-v1", model_kwargs={"device": "cuda"}
    )
    return FAISS.load_local("vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")


@st.cache_resource
def get_llm_openai():
    return ChatOpenAI(model_name="gpt-4o", **generation_config)


@st.cache_resource
def get_llm_reranking():
    return ChatOpenAI(model_name="gpt-4o", **generation_config)


@st.cache_resource
def get_streaming_llm_openai():
    return ChatOpenAI(model_name="gpt-4o", streaming=True, **generation_config)


@st.cache_resource
def get_llm_llama():
    llm = HuggingFaceEndpoint(
    endpoint_url=f"{hg_endpoint_url}",
    max_new_tokens=512,
    temperature=0.01,
    top_p=0.85,
    typical_p=0.95,
    repetition_penalty=1.2,
    return_full_text = False,
    stop_sequences = ['assistant'],
    )
    return ChatHuggingFace(llm=llm)


@st.cache_resource
def get_streaming_llm_llama():
    llm = HuggingFaceEndpoint(
    endpoint_url=f"{hg_endpoint_url}",
    max_new_tokens=512,
    temperature=0.01,
    top_p=0.85,
    typical_p=0.95,
    repetition_penalty=1.2,
    return_full_text = False,
    stop_sequences = ['assistant']
    )
    return ChatHuggingFace(llm=llm, streaming=True)