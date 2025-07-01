import streamlit as st
from pathlib import Path
from populate_database import populate
from query_data import get_source_filenames, query_rag

# ------------------- Konfigurasi halaman -------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------- CSS Kustom untuk Tampilan Kiri-Kanan -------------------
st.markdown("""
<style>
    /* Mengatur bubble chat */
    .st-emotion-cache-4oy321 {
        width: 100%;
        margin-bottom: 1rem;
    }
    /* Bubble untuk AI (kanan) */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) {
        display: flex;
        justify-content: flex-end;
    }
    /* Bubble untuk User (kiri) */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) {
        display: flex;
        justify-content: flex-start;
    }
    /* Konten di dalam bubble */
    .st-emotion-cache-1c7y2kd > div {
        max-width: 70%;
        padding: 1rem;
        border-radius: 10px;
        word-wrap: break-word;
    }
    /* Warna bubble AI */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) .st-emotion-cache-1c7y2kd > div {
        background-color: #2d2f31;
    }
    /* Warna bubble User */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) .st-emotion-cache-1c7y2kd > div {
        background-color: #3a3b3c;
    }
    /* Header dan status */
    .main-header { text-align: center; padding: 1rem 0; margin-bottom: 2rem; }
    .status-success { color: #00c37a; font-weight: bold; }
    .status-error { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------- Helper -------------------
def get_database_info():
    """Mengecek apakah database sudah ada."""
    return Path("chroma").exists()

@st.cache_data
def load_source_names_from_db():
    """Memuat nama file dari database dan meng-cache hasilnya."""
    return get_source_filenames()

# ------------------- State Init -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemma:2b" # Model default

# ------------------- Header -------------------
st.markdown('<div class="main-header"><h1>🤖 RAG Assistant</h1><p>Pilih model dan dokumen, lalu tanyakan apapun tentang isinya.</p></div>', unsafe_allow_html=True)

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")

    # Dropdown untuk memilih model
    model_options = ["gemma:2b", "phi", "deepseek-r1:1.5b"]
    st.session_state.selected_model = st.selectbox(
        label="Pilih model AI",
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )

    st.markdown("---")
    st.header("✍️ Pilih Dokumen Aktif")

    if get_database_info():
        st.markdown('<p class="status-success">✅ Database siap</p>', unsafe_allow_html=True)
        
        all_source_names = load_source_names_from_db()
        options = [name.split('/')[-1] for name in all_source_names]
        
        selected_files = st.multiselect(
            label="Pilih dokumen yang akan digunakan sebagai sumber jawaban.",
            options=options,
            key="selected_files_state"
        )
        
        selected_full_paths = [
            source for source in all_source_names 
            if source.split('/')[-1] in selected_files
        ]

        if selected_full_paths:
            st.success(f"{len(selected_full_paths)} dokumen aktif.")
        else:
            st.info("Tidak ada dokumen aktif. AI akan menjawab dengan pengetahuan umum.")
    else:
        st.markdown('<p class="status-error">❌ Database belum siap</p>', unsafe_allow_html=True)

    # Opsi Upload dan Reset
    st.markdown("---")
    st.header("📁 Kelola Dokumen")
    uploaded_files = st.file_uploader("Upload PDF baru", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("🚀 Proses Dokumen", use_container_width=True):
            data_path = Path("data"); data_path.mkdir(exist_ok=True)
            for uploaded_file in uploaded_files:
                 with open(data_path / uploaded_file.name, "wb") as f: f.write(uploaded_file.getbuffer())
            with st.spinner("Memproses PDF ke database..."): populate()
            st.cache_data.clear(); st.rerun()

    if st.button("🗑️ Reset Database", use_container_width=True):
        with st.spinner("Mereset database..."): populate(reset=True)
        st.cache_data.clear(); st.rerun()

# ------------------- Main Chat -------------------
st.header("💬 Chat")

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.spinner(f"🤖 Model '{st.session_state.selected_model}' sedang berpikir..."):
        result = query_rag(prompt, selected_full_paths, st.session_state.selected_model)
        full_response = result["answer"]
        
        if result["sources"]:
            sources_list = sorted(list(set(src.split('/')[-1] for src in result["sources"])))
            sources_text = "\n\n*Sumber: " + ", ".join(sources_list) + "*"
            full_response += sources_text
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(full_response)

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaaaaa; padding: 1rem;">
    <small>Dibuat dengan menggunakan Streamlit, LangChain & Ollama</small>
</div>
""", unsafe_allow_html=True)