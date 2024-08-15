import pandas as pd
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings( 
    model_name="multi-qa-mpnet-base-dot-v1", model_kwargs={"device": "gpu"}
)


def retrieval_score(df, vectorstore):
    results = []
    count = 0
    
    for _, row in df.iterrows():
        question = row['question']
        ground_truth_url = row['url']
        
        docs = vectorstore.similarity_search_with_score(query=question, k=100)
        top_8_urls = [doc.metadata.get('url') for doc, _ in docs[:5]]
       
        if ground_truth_url in top_8_urls:
            result = 1
            count += 1
        else:
            result = 0
        
        results.append({
            'question': question,
            'ground_truth_url': ground_truth_url,
            'top_8_urls': top_8_urls,
            'result': result
        })
    
    # Lưu kết quả vào file CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('labeled_retrieval_data.csv', index=False)
    retrieval_score = count / len(results_df)
    return retrieval_score

vectorstore = FAISS.load_local("vectorstore", embeddings=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":100})

df = pd.read_csv("data_test.csv")
score = retrieval_score(df, vectorstore)
print(score)