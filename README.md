
# AI Sales v3

## Run
pip install -r requirements.txt

### Setup Qdrant
python qdrant_setup.py

### Run API
uvicorn api:app --reload

### Run pipeline
python pipeline.py

## Cron (Linux)
crontab -e

0 2 * * 1 python /path/pipeline.py

## Flow
Excel -> pipeline -> Qdrant -> API -> LLM
