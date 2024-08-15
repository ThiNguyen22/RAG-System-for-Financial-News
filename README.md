# RAG-System-for-Financial-News


Training the RAG model: Combining financial information retrieval from databases with text generation from a large language model (Llama-3-8B) to answer questions related to financial information.

Using the Chain of Note method to improve RAG when dealing with noisy and irrelevant documents, while also addressing "unknown" situations.


## Acknowledgements

 - [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
 - [CHAIN-OF-NOTE: ENHANCING ROBUSTNESS IN RETRIEVAL-AUGMENTED LANGUAGE MODELS](https://arxiv.org/pdf/2311.09210)
 - [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/html/2309.01431v2)


## Installation and Setup


The requirements.txt file lists all the required Python libraries and they will be installed using:
```bash
  pip install -r requirements.txt
```

    
## Environment Variables
HUGGINGFACEHUB_API_TOKEN

OPENAI_API_KEY
### Windows Set-up

#### Option 1: Set your "HUGGINGFACEHUB_API_TOKEN" Environment Variable via the cmd prompt

Run the following in the cmd prompt, replacing <yourkey> with your API key:
```bash
setx HUGGINGFACEHUB_API_TOKEN "<yourkey>"
```
This will apply to future cmd prompt window, so you will need to open a new one to use that variable with curl. You can validate that this variable has been set by opening a new cmd prompt window and typing in 
```bash
echo %HUGGINGFACEHUB_API_TOKEN%
```
#### Option 2: Set your 'HUGGINGFACEHUB_API_TOKEN' Environment Variable through the Control Panel

    1. Open System properties and select Advanced system settings
    2. Select Environment Variables...
    3. Select New… from the User variables section(top). Add your name/key value pair, replacing <yourkey> with your API key.
```bash
Variable name: HUGGINGFACEHUB_API_TOKEN
Variable value: <yourkey>
```
Do the same with "OPENAI_API_KEY" !



## Run app

```bash
streamlit run app.py
```
## Detailed Overview
### 1. Architecture of Retrieval Augmented Generation (RAG)

![RAG_flow](https://github.com/Maithaoly/RAG-System-for-Finanical-News/assets/90881432/ed6ecf09-e9ba-4f18-b16e-bf8f3161316f)

The operation process of RAG can be summarized as follows:

•  Create Vector database: Initially, convert the entire knowledge data into vectors and store them in a vector database.

•  User enters a question: The user provides a query in natural language to search for an answer or to complete that query.

•  Information retrieval: The retrieval mechanism scans all vectors in the database to identify which knowledge segments (paragraphs) are semantically similar to the user's query. These paragraphs are then fed into the LLM to enhance the context for the answer generation process.

•  Combining data: The paragraphs retrieved from the database are combined with the original user query to create a prompt.

•  Generate text: The prompt, now enriched with additional context, is passed through the LLM to generate the final response according to the added context.

### 2. Chain of Note
The core idea of CoN is to create sequential reading notes for retrieved documents, allowing a comprehensive assessment of their relevance to the posed question and integrating this information to provide the final answer.

•  When the document directly answers the question, the model uses this information to provide the final answer.

•  If the document does not directly answer but provides useful context, the model will use this information along with inherent knowledge to infer the answer.

•  In cases where the documents are irrelevant and the model lacks the knowledge to answer, it will respond with "unknown".

#### 2.1 Notes design
![image](https://github.com/Maithaoly/RAG-System-for-Finanical-News/assets/90881432/df726c7a-af35-455b-b530-dac068d0de49)

 Illustration of the CHAIN-OF-NOTE (CON) framework with three distinct types of reading notes. 

Type (a) depicts the scenario where the language model identifies a document that directly answers the query, leading to a final answer formulated from the retrieved information. 

Type (b) represents situations where the retrieved document, while not directly answering the query, provides
contextual insights, enabling the language model to integrate this context with its inherent knowledge
to deduce an answer. 

Type (c) illustrates instances where the language model encounters irrelevant documents and lacks the necessary knowledge to respond, resulting in an “unknown” answer. 

This figure exemplifies the CoN framework’s capability to adaptively process information, balancing direct information retrieval, contextual inference, and the recognition of its knowledge boundaries.

#### 2.2 Data collection
![image](https://github.com/user-attachments/assets/f5425115-bc14-48d3-b568-b0a1bb1ee141)

To equip the model with the ability to generate reading notes, data collection for training is necessary. Creating reading notes manually is resource-intensive, so the GPT-4 language model is used to generate note data. First, nearly 1K questions from the pre-built QA set are used to retrieve five paragraphs from the database; these paragraphs are collectively referred to as the context. Then, GPT-4 is prompted with specific instructions to combine this context with the corresponding questions. The prediction quality of GPT-4 is assessed by us on a small set before proceeding with the entire dataset.

#### 2.3 Fine-tuning Process
![image](https://github.com/user-attachments/assets/ce99399e-0a3b-4bf0-9cae-8806459ce3bd)

To train the model to generate CoN, we use the fine-tuning technique. After collecting the training data from ChatGPT, the next step is to use it to train the CoN model, based on the open-source model Llama3 8B. With the input being a combination of instructions, context, and answers generated by ChatGPT, the model is then trained to produce notes and answers. The goal of this technique is for the model to understand and execute instructions more accurately and effectively.

### 3. The RAG system equipped with CoN for retrieving financial news
![image](https://github.com/user-attachments/assets/f6acb256-382c-44d9-873a-c8a8c57ac0e2)

The system takes in the chat history and the question, then responds to the question. The retrieval process with the LLM, which is the Llama 3 8B model equipped with CoN, is as follows:

  1. Use the chat history and the new question to create a "standalone question." This is done so that this question can be used in the retrieval step to fetch relevant documents. If only the new question is used, the relevant context might be missing. If the entire conversation is used in retrieval, it might include unnecessary information that could distract from the retrieval process.

  2. The new question is sent to the retrieval tool, and a list of 100 documents potentially related to the question is returned. However, the retrieval system might fetch documents that are not actually relevant to the search query.

  3. A re-ranker based on a cross-encoder is used to evaluate the relevance of the 100 documents to the search query and apply a time decay technique. The result is a ranked list of the top five most relevant documents.

  4. Instructions, the retrieved documents, and the new question are fed into the prompt, which is then passed to the large language model (LLM) to generate the final answer.


## Demo

[streamlit-app-2024-07-22-15-07-13.webm](https://github.com/user-attachments/assets/63b832e9-8d93-4571-b581-2356a19768d2)


## Feedback

If you have any feedback, please reach out to us at nguyenthihongthi.230502@gmail.com

