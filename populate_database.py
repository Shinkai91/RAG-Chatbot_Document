import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma")

def populate(reset: bool = False):
    if reset and CHROMA_DIR.exists():
        print("🗑️ Menghapus database Chroma lama...")
        shutil.rmtree(CHROMA_DIR)

    if not DATA_DIR.exists():
        raise FileNotFoundError("📂 Folder data/ tidak ditemukan. Upload PDF terlebih dahulu.")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("❗ Tidak ada file PDF di dalam folder data/")

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

    print(f"📄 Total dokumen ditemukan: {len(all_documents)}")

    # Split dokumen
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(all_documents)

    print(f"🔪 Total dokumen setelah split: {len(split_docs)}")

    # Embedding dan simpan ke vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    vectordb.persist()
    print("✅ Database berhasil dibuat dan disimpan!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset database")
    args = parser.parse_args()
    populate(reset=args.reset)