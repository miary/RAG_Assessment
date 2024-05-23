import os
from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2", "trust_remote_code=True")
#print(amnesty_qa)


# 1. Faithfulness - Measures the factual consistency of the answer to the context based on the question.
# 2. Context_precision - Measures how relevant the retrieved context is to the question, conveying the quality of the retrieval pipeline.
# 3. Answer_relevancy - Measures how relevant the answer is to the question.
# 4. Context_recall - Measures the retriever’s ability to retrieve all necessary information required to answer the question.

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ]
)

df = result.to_pandas()
print(df.head())
df.to_csv('synthetic_rageval.csv')
