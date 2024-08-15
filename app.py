import streamlit as st
import os
import json

from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import ChatMessage
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains import LLMChain

from prompt_default import *
from handlers import *
from models import *
from utils import *


def initialize_langchain_components():
    vectorstore = get_vector_store()
    cross_encoder = get_cross_encoder()
    llm = get_llm_llama()
    streaming_llm = get_streaming_llm_llama()
    llm_reranking = get_llm_reranking()
    llm_filter = LLMChainFilter.from_llm(llm_reranking)#, prompt=PROMPT_FILTER())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    re_ranking_compressor = ReRankingCompressor(
        cross_encoder=cross_encoder, topk=5, llm_chain_ranking=llm_filter
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=re_ranking_compressor, base_retriever=retriever
    )

    return llm, streaming_llm, compression_retriever


def generate_simple_answer(streaming_llm, prompt_, stream_handler):
    simple_chain = LLMChain(
        llm=streaming_llm,
        prompt = SIMPLE_PROMPT(),
        callbacks=[stream_handler],
        verbose=False,
    )
    result = simple_chain.run(
        {"question": prompt_}, callbacks=[stream_handler]
    )
    return result


def generate_answer(llm, streaming_llm, compression_retriever, prompt_, stream_handler):
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT())
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT()
    )
    qa_chain = ConversationalRetrievalChain(
        retriever=compression_retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )
    chat_history = [
        (msg_user.content, msg_ai.content)
        for msg_user, msg_ai in zip(
            *[iter(st.session_state.messages[1:])] * 2
        )
    ]

    result = qa_chain.run(
        {
            "question": prompt_,
            "chat_history": chat_history,
        },
        callbacks=[stream_handler],
    )
    return result


def process_assistant_response(llm, streaming_llm, compression_retriever, prompt_):
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        if len(prompt_.split()) < 3:
            result = generate_simple_answer(streaming_llm, prompt_, stream_handler)
        else:
            result = generate_answer(llm, streaming_llm, compression_retriever, prompt_, stream_handler)

        st.session_state.messages.append(ChatMessage(role="assistant", content=result))
        save_chat_history()


def process_user_message(llm, streaming_llm, compression_retriever, prompt_):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt_))
    st.chat_message("user").write(prompt_)
    process_assistant_response(llm, streaming_llm, compression_retriever, prompt_)


def handle_user_input(llm, streaming_llm, compression_retriever):
    if prompt_ := st.chat_input():
        process_user_message(llm, streaming_llm, compression_retriever, prompt_)


def save_chat_history():
    with open(os.path.join("history_chat", f"{st.session_state.session_id}.json"), "w") as f:
        json.dump({"data": [i.dict() for i in st.session_state.messages]}, f)


def initialize_session_state():
    if "messages" not in st.session_state:
        handle_new_session()


def display_chat_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def check_read_only(llm, streaming_llm, compression_retriever):
    read_only = st.session_state.get("read_only", False)
    if not read_only:
        handle_user_input(llm, streaming_llm, compression_retriever)


def main():
    
    setup_streamlit()
    create_sidebar_expander()
    llm, streaming_llm, compression_retriever = initialize_langchain_components()
    initialize_session_state()
    display_chat_messages()
    check_read_only(llm, streaming_llm, compression_retriever)


if __name__ == "__main__":
    main()