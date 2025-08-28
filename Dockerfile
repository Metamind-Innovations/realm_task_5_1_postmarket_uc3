FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "post_market_evaluation.py", \
             "--original_samples_path", "data/original_samples", \
             "--synthetic_samples_path", "data/synthetic_samples", \
             "--expert_knowledge_results_path", "data/expert_knowledge_results.json", \
             "--statistical_analysis_results_path", "data/statistical_analysis_results.json"]