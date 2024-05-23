## RAG Evaluation using RAGAS

### Assess RAG performance using synthetic data:
- Use a generator LLM and a critic LLM
- Generate reports:
  - answer relevancy
  - faithfulness
  - context recall
  - context precision

### Setup:
- Create a folder, 'docs', and place documents in folder
- Create the 'models/llm' and 'models/embedding' folders
- Download and place the GGUF LLM and Embedding models in their respective folder
- Copy env.example to .env and update the OpenAI key value
- Create a python virtual environment: python -m venv .venv
- Activate environment: .venv\Scripts\activate (Windows) and source .venv/bin/activate (Mac)
- Run pip install -r requirements.txt
  