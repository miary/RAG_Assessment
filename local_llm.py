from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
# default extractor
from ragas.testset.extractor import KeyphraseExtractor
from langchain.text_splitter import TokenTextSplitter
# default DocumentStore
from ragas.testset.docstore import InMemoryDocumentStore

from langchain.llms import LlamaCpp


llm_cpp = LlamaCpp(
            streaming = True,
            model_path="/content/drive/MyDrive/LLM_Model/zephyr-7b-beta.Q4_K_M.gguf",
            n_gpu_layers=2,
            n_batch=512,
            temperature=0.75,
            top_p=1,
            verbose=True,
            n_ctx=4096
            )


# define llm and embeddings
langchain_llm = BaseLanguageModel(model=llm_cpp) # any langchain LLM instance
langchain_embeddings = Embeddings(model="my_model") # any langchain Embeddings instance

# make sure to wrap them with wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

langchain_llm = LangchainLLMWrapper(langchain_llm)
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# Can also use custom LLMs and Embeddings here but make sure 
# they are subclasses of BaseRagasLLM and BaseRagasEmbeddings
llm = langchain_llm # MyCustomLLM()
embeddings = langchain_embeddings # MyCustomEmbeddings()


# init the DocumentStore with own llm and embeddings
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=langchain_embeddings,
    extractor=keyphrase_extractor,
)

