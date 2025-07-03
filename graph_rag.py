from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from typing import TypedDict, List, Optional
from langgraph.graph import END, StateGraph

# --- Konfigurasi Awal ---
CHROMA_DIR = "chroma"

class GraphState(TypedDict):
    """
    Mendefinisikan state dari graph. Setiap field adalah bagian dari "memori"
    yang bisa diakses dan dimodifikasi oleh setiap node.
    """
    question: str
    generation: str
    documents: List[Document]
    model_name: str
    selected_sources: List[str]
    run_rag: Optional[str]

def retrieve(state: GraphState) -> GraphState:
    """
    Node untuk mengambil dokumen dari database vektor (ChromaDB).
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]
    model_name = state["model_name"]
    selected_sources = state["selected_sources"]

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function
    )
    
    filter_dict = {"source": {"$in": selected_sources}} if selected_sources else {}

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'filter': filter_dict, 'k': 10}
    )
    documents = retriever.invoke(question)
    
    print(f"Retrieved {len(documents)} documents.")
    
    return {
        "documents": documents,
        "question": question
    }

def grade_documents(state: GraphState) -> GraphState:
    """
    Node untuk mengevaluasi relevansi dokumen yang telah diambil.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    model_name = state["model_name"]
    
    if not documents:
        print("No documents retrieved, deciding to generate without context.")
        return {"documents": documents, "question": question, "run_rag": "no"}

    llm = Ollama(model=model_name, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""Anda adalah seorang penilai yang bertugas menentukan apakah sekumpulan dokumen relevan dengan pertanyaan pengguna.
        Berikut adalah dokumen yang diambil (dipisahkan oleh '---'):
        {context}
        Berikut adalah pertanyaan pengguna: {question}
        Apakah dokumen-dokumen ini mengandung informasi yang cukup untuk menjawab pertanyaan tersebut?
        Berikan jawaban dalam format JSON dengan kunci "score" yang bernilai "yes" atau "no".
        Contoh: {{"score": "yes"}} """,
        input_variables=["context", "question"],
    )

    chain = prompt | llm | JsonOutputParser()
    
    docs_content = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    try:
        result = chain.invoke({"context": docs_content, "question": question})
        grade = result.get('score', 'no')
    except Exception as e:
        print(f"Error during grading: {e}")
        grade = "no"
    
    if grade == "yes":
        print("GRADE: Dokumen relevan, lanjut ke generate.")
        return {"run_rag": "yes"}
    else:
        print("GRADE: Dokumen tidak relevan, coba tulis ulang pertanyaan.")
        return {"run_rag": "no"}

def generate(state: GraphState) -> GraphState:
    """
    Node untuk menghasilkan jawaban berdasarkan pertanyaan dan dokumen yang relevan.
    """
    print("---NODE: GENERATE (WITH RAG)---")
    question = state["question"]
    documents = state["documents"]
    model_name = state["model_name"]
    
    llm = Ollama(model=model_name, temperature=0.2)
    
    prompt = PromptTemplate(
        template="""Anda adalah asisten AI yang cerdas. Gunakan potongan konteks berikut untuk menjawab pertanyaan.
        Jika Anda tidak tahu jawabannya dari konteks yang diberikan, katakan saja Anda tidak tahu. Jawab dengan ringkas dan jelas.

        Konteks: {context}
        Pertanyaan: {question}
        Jawaban:""",
        input_variables=["context", "question"],
    )
    
    docs_content = "\n\n".join([doc.page_content for doc in documents])
    
    # --- PERUBAHAN 2: Tambahkan StrOutputParser ---
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": docs_content, "question": question})
    
    return {
        "generation": generation,
        "documents": documents
    }

def rewrite_query(state: GraphState) -> GraphState:
    """
    Node untuk menulis ulang pertanyaan pengguna agar lebih baik untuk pencarian.
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    model_name = state["model_name"]

    llm = Ollama(model=model_name, temperature=0)
    
    prompt = PromptTemplate(
        template="""Anda adalah asisten AI yang ahli dalam menyempurnakan query.
        Tulis ulang pertanyaan berikut agar lebih mudah dicari di database vektor.
        Fokus pada kata kunci utama dan maksud dari pertanyaan.
        
        Pertanyaan asli: {question}
        Pertanyaan yang disempurnakan:""",
        input_variables=["question"],
    )
    
    # --- PERUBAHAN 3: Tambahkan StrOutputParser ---
    rewrite_chain = prompt | llm | StrOutputParser()
    
    rewritten_question = rewrite_chain.invoke({"question": question})
    
    print(f"Rewritten question: {rewritten_question}")
    
    return {"question": rewritten_question}

def fallback(state: GraphState) -> GraphState:
    """
    Node fallback jika RAG tidak berhasil (dokumen tidak relevan).
    """
    print("---NODE: FALLBACK (NO RAG)---")
    question = state["question"]
    model_name = state["model_name"]

    llm = Ollama(model=model_name, temperature=0.2)
    
    prompt = PromptTemplate(
        template="""Anda adalah asisten AI yang membantu. Jawab pertanyaan berikut dengan pengetahuan umum Anda. 
                 Katakan bahwa Anda menjawab tanpa menggunakan dokumen karena tidak ada yang relevan.

                 Pertanyaan: {question}
                 Jawaban:""",
        input_variables=["question"],
    )
    
    # --- PERUBAHAN 4: Tambahkan StrOutputParser ---
    fallback_chain = prompt | llm | StrOutputParser()
    
    generation = fallback_chain.invoke({"question": question})
    
    return {
        "generation": generation,
        "documents": [] 
    }

def decide_to_generate(state: GraphState) -> str:
    """
    Memutuskan apakah akan melanjutkan ke tahap 'generate' atau 'rewrite'.
    """
    print("---DECISION: GENERATE OR REWRITE---")
    
    if state.get("run_rag") == "yes":
        return "generate"
    else:
        if not state.get("selected_sources"):
            print("No sources selected, going to fallback.")
            return "fallback"
        else:
            print("Documents not relevant, going to rewrite.")
            return "rewrite"

def build_and_run_graph(query_text: str, selected_sources: list[str], model_name: str) -> dict:
    """
    Membangun, mengompilasi, dan menjalankan graph LangGraph.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("fallback", fallback)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()

    inputs = {
        "question": query_text,
        "model_name": model_name,
        "selected_sources": selected_sources
    }
    result = app.invoke(inputs)

    raw_answer = result.get("generation", "Tidak dapat menghasilkan jawaban.").strip()
    sources = [doc.metadata.get("source", "Tidak diketahui") for doc in result.get("documents", [])]
    unique_sources = sorted(list(set(sources)))

    return {
        "answer": raw_answer,
        "sources": unique_sources
    }

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