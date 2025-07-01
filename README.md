# 🤖 RAG Assistant - Asisten AI dengan Konteks Dokumen

Selamat datang di **RAG Assistant**! Aplikasi web ini memungkinkan Anda berinteraksi dengan model bahasa (LLM) yang kemampuannya ditingkatkan dengan pengetahuan dari dokumen PDF yang Anda unggah. Pilih dokumen sebagai "otak" tambahan untuk AI, pilih model AI yang ingin digunakan, lalu ajukan pertanyaan apa pun terkait isi dokumen tersebut.

Aplikasi ini dibangun menggunakan arsitektur **RAG (Retrieval-Augmented Generation)**, yang menggabungkan kekuatan pencarian informasi presisi dengan kemampuan generasi teks dari LLM.

---

## ✨ Fitur Utama

- **Unggah PDF Fleksibel**: Unggah satu atau lebih dokumen PDF untuk dijadikan basis pengetahuan.
- **Pemilihan Konteks Dinamis**: Pilih dan ganti dokumen mana yang ingin dijadikan konteks aktif langsung dari antarmuka.
- **Dukungan Multi-Model**: Ganti model AI (seperti Gemma, Phi, DeepSeek) secara *real-time* untuk membandingkan jawaban.
- **Pencarian Cerdas**: Menggunakan metode *Maximal Marginal Relevance (MMR)* untuk menemukan informasi paling relevan dari dokumen Anda.
- **Antarmuka Intuitif**: Dibuat dengan Streamlit, memberikan pengalaman pengguna yang bersih dan responsif.
- **Lokal dan Privat**: Semua model dan data berjalan di mesin Anda sendiri menggunakan Ollama, memastikan privasi data Anda sepenuhnya.

---

## 🛠️ Tumpukan Teknologi (Tech Stack)

Proyek ini dibangun menggunakan beberapa teknologi dan pustaka Python berikut:

* **Framework Aplikasi**: [Streamlit](https://streamlit.io/)
* **Orkestrasi LLM**: [LangChain](https://www.langchain.com/)
* **Model Bahasa (LLM)**: [Ollama](https://ollama.com/) (Menjalankan model seperti Gemma, Phi, dll. secara lokal)
* **Database Vektor**: [ChromaDB](https://www.trychroma.com/)
* **Embedding Model**: `nomic-embed-text`
* **PDF Loader**: `PyPDFLoader`
---

## ⚙️ Instalasi dan Pengaturan

Ikuti langkah-langkah berikut untuk menjalankan proyek di komputer Anda.

### 1. Prasyarat

Pastikan Anda sudah menginstal **Python 3.8+** dan **Ollama**.

- **Instal Ollama**: Kunjungi [situs web Ollama](https://ollama.com/) dan ikuti petunjuk instalasi untuk sistem operasi Anda (macOS, Linux, Windows).

### 2. Unduh Model AI yang Dibutuhkan

Buka terminal atau command prompt, lalu jalankan perintah berikut untuk mengunduh model-model yang diperlukan:

```bash
ollama pull gemma:2b
ollama pull phi
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
```

### 3. Siapkan Proyek

#### a. Clone atau Unduh Proyek

- Jika menggunakan Git, clone repositori ini.
- Jika tidak, unduh dan ekstrak file `main.py`, `query_data.py`, dan `populate_database.py` ke dalam satu folder.

#### b. Buat Virtual Environment (Sangat Direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate
```

#### c. Instal Pustaka Python yang Dibutuhkan

Buat file `requirements.txt` dengan isi berikut:

```
streamlit
langchain
langchain-community
langchain-core
chromadb
pypdf
ollama
```

Lalu instal semua dependensi:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Penggunaan

Setelah instalasi selesai, ikuti tiga langkah mudah ini untuk menggunakan aplikasi.

### 1. Siapkan Folder Data

Buat folder bernama `data` di dalam direktori utama proyek Anda. Tempatkan semua file PDF yang ingin dianalisis di folder ini.

```
/proyek-anda/
|-- 📂 data/
|   |-- laporan_keuangan.pdf
|   |-- manual_produk.pdf
|-- main.py
|-- query_data.py
|-- populate_database.py
|-- requirements.txt
```

### 2. Buat Database Vektor

Jalankan skrip `populate_database.py` dari terminal untuk membaca semua PDF di folder `data`, memecahnya menjadi potongan kecil (*chunks*), dan menyimpannya ke dalam database vektor Chroma.

```bash
python populate_database.py
```

Lakukan ini setiap kali Anda menambah atau mengubah file PDF di folder `data`. Untuk membuat ulang database dari awal, gunakan flag `--reset`:

```bash
python populate_database.py --reset
```

### 3. Jalankan Aplikasi Streamlit

Jalankan aplikasi utama dengan perintah berikut:

```bash
streamlit run main.py
```

Aplikasi akan terbuka otomatis di browser web Anda.

---

### Navigasi Aplikasi

- **Pilih Model AI**: Gunakan dropdown di sidebar kiri untuk memilih model yang ingin digunakan.
- **Pilih Dokumen**: Pilih satu atau lebih file PDF sebagai konteks aktif.
- **Mulai Bertanya**: Ketik pertanyaan di kolom chat di bagian bawah dan tekan Enter. AI akan menjawab berdasarkan dokumen yang dipilih.
- **Kelola Dokumen**: Unggah PDF baru atau reset database langsung dari sidebar.