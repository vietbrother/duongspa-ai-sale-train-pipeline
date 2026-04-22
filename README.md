
# DuongSpa AI Sales Training Pipeline v3.1

Pipeline training tự động để tạo dataset chất lượng cao cho RAG và Fine-tuning.

## 📋 Tổng quan

Pipeline xử lý:
1. **Load data** — CRM, Conversations, Chatpage staff
2. **Join & Enrich** — Kết hợp data với CRM (phone matching)
3. **Feature Engineering** — Extract features (reward, score, segment)
4. **Ground Truth** — Label outcomes (won/booked/pending/lost)
5. **Tone & Style** — Extract phong cách sales từ top performers
6. **Chunking** — Tách conversations thành segments chất lượng
7. **Embedding** — Tạo vectors với OpenAI ada-002
8. **Export** — JSON, JSONL (fine-tune), embeddings
9. **Qdrant** — Push vectors vào Qdrant collection

**Output**:
- `finetune_conversations_*.jsonl` — Fine-tuning dataset (OpenAI format)
- `qa_pairs_*.json` — Q&A pairs cho RAG
- `style_profile_*.json` — Tone & style từ top sales
- `report_*.json` — Statistics & metrics
- Qdrant collection với vectors + metadata

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.9+**
- **Qdrant** (running on localhost:6333 hoặc cloud)
- **OpenAI API Key** (cho embedding)

### Ubuntu/Linux

#### 1. Clone repository
```bash
cd /path/to/project
```

#### 2. Tạo virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
Neu cpu cu qua thi phai ha python xuong v3.10
  206  sudo apt update
  207  sudo apt install software-properties-common
  208  sudo add-apt-repository ppa:deadsnakes/ppa
  209  sudo apt update
  210  sudo apt install python3.10 python3.10-venv python3.10-distutils
  211  python3.10 --version
  212  python3.10 -m venv venv310
  213  source venv310/bin/activate
  214  pip install --upgrade pip setuptools wheel


#### 3. Cài đặt dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Setup Qdrant (nếu chạy local)
```bash
# Cài Docker nếu chưa có
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker

# Chạy Qdrant với Docker
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Hoặc dùng Docker Compose (từ root project)
cd ..
docker-compose -f docker-compose.infra.yml up -d qdrant
```

#### 5. Cấu hình environment
```bash
# Copy file .env.example
cp .env.example .env

# Chỉnh sửa .env với editor yêu thích
nano .env  # hoặc vim, vi, gedit
```

Điền vào `.env`:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=conversation_memory
VECTOR_SIZE=1536

# Paths (relative hoặc absolute)
CRM_FILE=data-train/crm_customer_book_revenue_20260418.csv
CHAT_FILE=data-train/raw_message_convertation_20260418.xlsx
CHATPAGE_FILE=data-train/crm_chatpage_customer_20260416.csv
```

#### 6. Chuẩn bị data files
```bash
# Tạo thư mục data-train nếu chưa có
mkdir -p data-train

# Copy files từ data-train-format hoặc từ CRM export
cp data-train-format/*.csv data-train/
cp data-train-format/*.xlsx data-train/
```

### Windows

#### 1. Clone repository
```powershell
cd D:\Source\2026_04_11_duongspa_chatbot\duongspa-ai-sale-train-pipeline
```

#### 2. Tạo virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Nếu gặp lỗi Execution Policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Cài đặt dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Setup Qdrant (nếu chạy local)
```powershell
# Option 1: Docker Desktop
# - Cài Docker Desktop for Windows từ https://www.docker.com/products/docker-desktop
# - Chạy Docker Desktop
# - Mở PowerShell:

docker run -d -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage `
  qdrant/qdrant:latest

# Option 2: Docker Compose (từ root project)
cd ..
docker-compose -f docker-compose.infra.yml up -d qdrant
```

#### 5. Cấu hình environment
```powershell
# Copy file .env.example
Copy-Item .env.example .env

# Chỉnh sửa .env với Notepad hoặc VSCode
notepad .env
# hoặc
code .env
```

Điền vào `.env`:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=conversation_memory
VECTOR_SIZE=1536

# Paths (Windows - dùng forward slash hoặc escaped backslash)
CRM_FILE=data-train/crm_customer_book_revenue_20260418.csv
CHAT_FILE=data-train/raw_message_convertation_20260418.xlsx
CHATPAGE_FILE=data-train/crm_chatpage_customer_20260416.csv
```

#### 6. Chuẩn bị data files
```powershell
# Tạo thư mục data-train nếu chưa có
New-Item -ItemType Directory -Force -Path data-train

# Copy files từ data-train-format hoặc từ CRM export
Copy-Item data-train-format\*.csv data-train\
Copy-Item data-train-format\*.xlsx data-train\
```

---

## 🏃 Running the Pipeline

### Chạy Full Pipeline

**Ubuntu/Linux**:
```bash
# Activate virtual environment
source venv/bin/activate

# Chạy pipeline
python main.py
```

**Windows**:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Chạy pipeline
python main.py
```

### Output

Pipeline sẽ tạo thư mục `output/` với các file:
```
output/
├── finetune_conversations_20260423_143022.jsonl    # Fine-tuning dataset
├── qa_pairs_20260423_143022.json                   # Q&A pairs
├── style_profile_20260423_143022.json              # Tone & style
├── report_20260423_143022.json                     # Statistics
└── segments_20260423_143022.json                   # Raw segments
```

### Chạy API Server (Test mode)

**Ubuntu/Linux**:
```bash
source venv/bin/activate
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

**Windows**:
```powershell
.\venv\Scripts\Activate.ps1
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

Test API:
```bash
# Search similar conversations
curl "http://localhost:8001/search?q=giảm béo&segment=MID&top_k=3"

# Build chat prompt
curl "http://localhost:8001/chat?q=spa có dịch vụ gì&segment=HIGH"
```

---

## ⏰ Cron Job (Auto-run)

### Ubuntu/Linux

Chạy pipeline tự động mỗi tuần (2:00 AM Thứ 2):

```bash
# Mở crontab editor
crontab -e

# Thêm dòng sau:
0 2 * * 1 cd /path/to/duongspa-ai-sale-train-pipeline && /path/to/venv/bin/python main.py >> /var/log/duongspa_pipeline.log 2>&1

# Ví dụ cụ thể:
0 2 * * 1 cd /home/user/duongspa-ai-sale-train-pipeline && /home/user/duongspa-ai-sale-train-pipeline/venv/bin/python main.py >> /var/log/duongspa_pipeline.log 2>&1
```

**Giải thích**:
- `0 2 * * 1` — 2:00 AM mỗi Thứ 2
- `cd /path/...` — Di chuyển vào thư mục pipeline
- `/path/to/venv/bin/python` — Sử dụng Python từ virtual environment
- `>> /var/log/...` — Ghi log ra file
- `2>&1` — Redirect stderr vào stdout

### Windows (Task Scheduler)

#### Option 1: PowerShell Script

Tạo file `run_pipeline.ps1`:
```powershell
# run_pipeline.ps1
Set-Location "D:\Source\2026_04_11_duongspa_chatbot\duongspa-ai-sale-train-pipeline"
.\venv\Scripts\Activate.ps1
python main.py
```

Tạo Task Scheduler:
```powershell
# Mở Task Scheduler
taskschd.msc

# Hoặc dùng PowerShell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File D:\Source\2026_04_11_duongspa_chatbot\duongspa-ai-sale-train-pipeline\run_pipeline.ps1"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 2:00AM
Register-ScheduledTask -TaskName "DuongSpa Pipeline" -Action $action -Trigger $trigger -Description "Run AI Sales Training Pipeline weekly"
```

#### Option 2: Batch Script

Tạo file `run_pipeline.bat`:
```batch
@echo off
cd /d D:\Source\2026_04_11_duongspa_chatbot\duongspa-ai-sale-train-pipeline
call venv\Scripts\activate.bat
python main.py
pause
```

Thêm vào Task Scheduler thủ công:
1. Mở Task Scheduler (Win + R → `taskschd.msc`)
2. Create Basic Task → Tên: "DuongSpa Pipeline"
3. Trigger: Weekly, Monday, 2:00 AM
4. Action: Start a program → Browse: `run_pipeline.bat`
5. Finish

---

## 🔧 Configuration

### Environment Variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key cho embedding | **Required** |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `COLLECTION_NAME` | Qdrant collection name | `conversation_memory` |
| `VECTOR_SIZE` | Embedding dimension | `1536` (OpenAI ada-002) |
| `CRM_FILE` | Path to CRM CSV file | `data-train/crm_customer_book_revenue_*.csv` |
| `CHAT_FILE` | Path to conversations Excel file | `data-train/raw_message_convertation_*.xlsx` |
| `CHATPAGE_FILE` | Path to chatpage staff CSV | `data-train/crm_chatpage_customer_*.csv` |
| `SCORE_THRESHOLD` | Minimum quality score | `70` |
| `TOP_SALES_PERCENTILE` | Top sales percentile | `0.3` (top 30%) |
| `OUTPUT_DIR` | Output directory | `output/` |

### Config Constants (`config.py`)

- **Segment Rules**: Phân khúc khách hàng theo `paid_value`
- **State Keywords**: FSM states cho conversation states
- **Outcome Weights**: Trọng số cho ground truth labeling
- **Chunking**: `MAX_TURNS_PER_CHUNK`, `MIN_TURNS_PER_CHUNK`

---

## 📊 Pipeline Stages

### 1. **Load Data** (`loader.py`)
- Load CRM customer data
- Load conversation messages
- Load chatpage staff performance

### 2. **Join & Enrich** (`joiner.py`)
- Join conversations với CRM qua `phone`
- Enrich với booking_count, paid_value, status

### 3. **Feature Engineering** (`feature.py`)
- Extract features: `num_turns`, `turn_ratio`, `has_phone`, etc.

### 4. **Reward & Scoring** (`reward.py`, `scoring.py`)
- Compute reward score (lead capture, conversion, revenue)
- Compute quality score (conversation chất lượng)

### 5. **Segmentation** (`segment.py`)
- Phân khúc khách hàng: LOW, MID, HIGH, VIP

### 6. **Ground Truth** (`ground_truth.py`)
- Label outcomes: won, booked, pending, lost
- Apply outcome weights cho training

### 7. **Tone & Style Extraction** (`tone_style.py`)
- Extract phong cách sales từ top performers
- Emoji usage, greeting/closing patterns, CTA phrases

### 8. **Chunking** (`chunker.py`)
- Tách conversations thành segments (10-turn chunks)
- Detect conversation states (FSM)
- Generate Q&A pairs

### 9. **Embedding** (`embedding.py`)
- Embed text với OpenAI `text-embedding-ada-002`
- Retry logic + exponential backoff

### 10. **Qdrant Push** (`qdrant_client.py`)
- Create collection với vector config
- Push segments + metadata vào Qdrant

---

## 📁 Data Format

### Input Files

#### CRM File (`crm_customer_book_revenue_*.csv`)
```csv
phone,booking_count,total_paid_value,total_package_value,status,service_interest
0912345678,5,12000000,15000000,active,Giảm béo
```

#### Conversations File (`raw_message_convertation_*.xlsx`)
```
Mã tin nhắn | Nội dung tin nhắn | Tên người gửi | Số điện thoại | Tạo lúc | ...
msg_001     | Chào chị          | DƯỠNG         | 0912345678    | 2026-04-18 10:00 | ...
```

#### Chatpage Staff File (`crm_chatpage_customer_*.csv`)
```csv
name,total_phone
Nhân viên A,150
Nhân viên B,120
```

### Output Files

#### Fine-tuning Dataset (`finetune_conversations_*.jsonl`)
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [...]}
```

#### Q&A Pairs (`qa_pairs_*.json`)
```json
[
  {
    "text": "Khách: Spa có dịch vụ gì?\nSale: Dạ chúng em có...",
    "response": "Dạ chúng em có...",
    "segment": "MID",
    "score": 85.5
  }
]
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'xxx'`
**Solution**:
```bash
# Ubuntu
source venv/bin/activate
pip install -r requirements.txt

# Windows
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: data-train/*.csv not found`
**Solution**:
```bash
# Kiểm tra đường dẫn trong .env
# Đảm bảo files tồn tại trong data-train/
ls data-train/  # Ubuntu
dir data-train\  # Windows
```

### Issue: `qdrant_client.UnexpectedResponse: <Response [404]>`
**Solution**:
```bash
# Kiểm tra Qdrant đang chạy
curl http://localhost:6333/collections  # Ubuntu
Invoke-WebRequest http://localhost:6333/collections  # Windows

# Restart Qdrant nếu cần
docker restart <qdrant-container-id>
```

### Issue: `openai.AuthenticationError: Invalid API key`
**Solution**:
```bash
# Kiểm tra API key trong .env
cat .env | grep OPENAI_API_KEY  # Ubuntu
Select-String -Path .env -Pattern "OPENAI_API_KEY"  # Windows

# Update API key
# Lấy key mới từ https://platform.openai.com/api-keys
```

### Issue: Windows - `python: command not found`
**Solution**:
```powershell
# Kiểm tra Python đã cài đặt
python --version

# Nếu chưa có, download từ https://www.python.org/downloads/
# Hoặc dùng Microsoft Store
```

---

## 🔄 Pipeline Flow Diagram

```
┌─────────────────┐
│  CRM Data       │
│  (CSV)          │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
┌────────▼────────┐  ┌────▼──────────┐
│ Conversations   │  │ Chatpage      │
│ (Excel)         │  │ Staff (CSV)   │
└────────┬────────┘  └────┬──────────┘
         │                │
         └────────┬───────┘
                  │
         ┌────────▼────────┐
         │  Join & Enrich  │
         │  (phone match)  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Feature Extract │
         │ Reward/Score    │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Ground Truth    │
         │ Label Outcomes  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Tone & Style    │
         │ (Top Sales)     │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Chunking      │
         │ (Segments+Q&A)  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Embedding     │
         │ (OpenAI ada-002)│
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Export Files   │
         │ (JSONL, JSON)   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Push to Qdrant  │
         │  (Vectors +     │
         │   Metadata)     │
         └─────────────────┘
```

---

## 📝 Development

### Project Structure
```
duongspa-ai-sale-train-pipeline/
├── main.py                 # Main pipeline orchestrator
├── config.py               # Configuration & constants
├── loader.py               # Data loading (CRM, Chat, Chatpage)
├── joiner.py               # Join conversations + CRM
├── feature.py              # Feature engineering
├── reward.py               # Reward scoring
├── scoring.py              # Quality scoring
├── segment.py              # Customer segmentation
├── ground_truth.py         # Outcome labeling
├── tone_style.py           # Tone & style extraction
├── chunker.py              # Conversation chunking
├── embedding.py            # OpenAI embedding
├── qdrant_client.py        # Qdrant operations
├── state_engine.py         # FSM state detection
├── prediction.py           # Revenue prediction
├── prompt_builder.py       # Prompt generation
├── retrieval.py            # RAG search
├── api.py                  # FastAPI server (test mode)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── data-train/             # Input data folder
```

### Adding New Features

1. Modify `config.py` for new constants
2. Add processing logic in relevant modules
3. Update `main.py` to include new step
4. Test locally before deployment

---

## 📞 Support

**Issues?** Check:
- `.env` configuration
- Data files exist in `data-train/`
- Qdrant is running (`docker ps`)
- OpenAI API key is valid
- Virtual environment is activated

**Logs**: Pipeline outputs detailed logs to console. Redirect to file for debugging:
```bash
# Ubuntu
python main.py > pipeline.log 2>&1

# Windows
python main.py > pipeline.log 2>&1
```

---

**Version**: v3.1  
**Last Updated**: April 23, 2026  
**License**: Proprietary — DuongSpa Internal Use Only
