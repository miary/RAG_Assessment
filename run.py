import os

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def load_documents():
    root_path = os.getcwd()
    docs_path = os.path.join(root_path, 'docs')
    loader = DirectoryLoader(docs_path)
    documents = loader.load()
    #print(documents)

    for document in documents:
        document.metadata['filename'] = document.metadata['source']

    return documents


def generate_save_testset(documents):
    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(documents, test_size=70, 
                                distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
                        )
    testset.to_pandas()
    print(testset.head())
    testset.to_csv('synthetic.csv')


def load_saved_testset(file_csv):
    pass



documents = load_documents()
generate_save_testset(documents)
