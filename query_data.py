from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

CHROMA_DIR = "chroma"

def get_source_filenames() -> list[str]:
    """Mengambil daftar nama file PDF unik dari database."""
    try:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
        all_metadata = vectordb.get(include=["metadatas"])['metadatas']
        if not all_metadata:
            return []
        unique_sources = sorted(list(set(meta['source'] for meta in all_metadata)))
        return unique_sources
    except Exception:
        return []

def query_rag(query_text: str, selected_sources: list[str], model_name: str) -> dict:
    """
    Fungsi utama untuk RAG dengan metode pencarian MMR yang lebih andal.
    """
    llm = Ollama(model=model_name)

    if not selected_sources:
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("Jawab pertanyaan ini: '{question}'"))
        result = chain.invoke({"question": query_text})
        return {"answer": result.get("text", "Error."), "sources": []}

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

    filter_dict = {"source": {"$in": selected_sources}}

    # --- PERUBAHAN UTAMA: MENGGUNAKAN METODE PENCARIAN "mmr" ---
    retriever = vectordb.as_retriever(
        search_type="mmr", # Mengubah jenis pencarian
        search_kwargs={'filter': filter_dict, 'k': 12} # Memberi lebih banyak kandidat untuk MMR
    )
    # -----------------------------------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query_text})
    
    raw_answer = result.get("result", "Tidak dapat menghasilkan jawaban.").strip()
    
    sources = [doc.metadata.get("source", "Tidak diketahui") for doc in result.get("source_documents", [])]
    unique_sources = sorted(list(set(sources)))

    return {
        "answer": raw_answer,
        "sources": unique_sources
    }