from langchain_core.prompts.prompt import PromptTemplate
from langchain.prompts import Prompt



def PROMPT_FILTER():
    prompt_template = """Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

    > Question: {question}
    > Context:
    >>>
    {context}
    >>>
    > Relevant (YES / NO):"""
    
    prompt = PromptTemplate(
    template=prompt_template,input_variables=["question", "context"],
    )
    return prompt


def SIMPLE_PROMPT():
    prompt_template = """Answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
    template=prompt_template, input_variables=["question"]
    )
    return prompt


def QA_PROMPT():
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def CONDENSE_QUESTION_PROMPT():
    prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    prompt = PromptTemplate.from_template(prompt_template)
    return prompt
