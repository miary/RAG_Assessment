from langchain.document_loaders import ArxivLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings


# Documents handling
papers = ["2310.13800", "2307.03109", "2304.08637", "2310.05657", "2305.13091", "2311.09476", "2308.10633", "2309.01431", "2311.04348"]

docs_to_merge = []

for paper in papers:
    loader = ArxivLoader(query=paper)
    docs_to_merge.append(loader)

all_loaders = MergedDataLoader(loaders=docs_to_merge)

all_docs = all_loaders.load()

for doc in all_docs:
    print(doc.metadata)



# Documents embedding
model_name = "BAAI/bge-large-en-v1.5"

encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                               chunk_overlap = 16,
                                               length_function=len)

docs = text_splitter.split_documents(all_docs)
print(f"\nNumber of chunks: {len(docs)}\n")

vectorstore = Chroma.from_documents(docs, hf_bge_embeddings)

# Sanity check - Largest chunk size should be less than 512
print(max([len(chunk.page_content) for chunk in docs]))

base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 5})

# Retriever will grab chunks that are relevant to query.
query = "What are the challenges in evaluating Retrieval Augmented Generation pipelines?"
relevant_docs = base_retriever.get_relevant_documents(query)

# Inspect retrieved documents
for doc in relevant_docs:
  print(doc.page_content)
  print('\n')




# Reference: https://deci.ai/blog/evaluating-rag-pipelines-using-langchain-and-ragas/
