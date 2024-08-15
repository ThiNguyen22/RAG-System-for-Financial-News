from glob import glob
from typing import Optional, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import (
    Sequence,
    Document,
    BaseDocumentCompressor,
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class ReRankingCompressor(BaseDocumentCompressor):
    cross_encoder: Any
    topk: int
    llm_chain_ranking: Optional[LLMChainFilter] = None
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        
        cross_inp = [[query, hit.page_content] for hit in documents]
        cross_scores = self.cross_encoder.predict(cross_inp)

        docs_score = zip(documents, cross_scores)
        docs_score = [(docs, score - 8 if "news" in docs.metadata.get("link", "") else score) for docs, score in docs_score]
        filtered_docs = list(
            filter(
                lambda x: x[1] > -6,
                sorted(docs_score, key=lambda x: x[1], reverse=True),
            )
        )
        result = [i[0] for i in filtered_docs[: self.topk]]
        if self.llm_chain_ranking is not None:
            result = self.llm_chain_ranking.compress_documents(
                documents=result, query=query
            )

        return result
