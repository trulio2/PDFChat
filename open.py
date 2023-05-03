import langchain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

OPENAI_API_KEY = ""


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def create_docs(name):
    doc_reader = PdfReader("./pdf/")

    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    docsearch = FAISS.from_texts(texts, embeddings)
    docsearch.save_local("./db/", name)

    return docsearch


def load_docs(name):
    docsearch = FAISS.load_local('./db/', embeddings, name)
    return docsearch


docsearch = create_docs('openai')

# docsearch = load_docs('openai')

query = ""  # Semantic search

docs = docsearch.similarity_search(query)

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo",
                 openai_api_key=OPENAI_API_KEY)

chain = load_qa_chain(llm, chain_type="stuff")

while True:
    query = input("Enter your question: ")
    print(chain.run(input_documents=docs, question=f"USER: {query}"))
