# Test — Qdrant Verification

Thư mục này chứa script kiểm tra dữ liệu sau khi pipeline embedding hoàn tất.

## Files

| File | Mô tả |
|---|---|
| `test_qdrant.sh` | Bash script (Linux/macOS/WSL) dùng `curl` — kiểm tra Qdrant API |
| `test_qdrant.ps1` | PowerShell script (Windows) — kiểm tra Qdrant API |
| `test_search_openai.py` | Python — test semantic search với **OpenAI** embedding |
| `test_search_local.py` | Python — test semantic search với **local** sentence-transformers |

---

## test_search_openai.py

Test toàn bộ pipeline search với OpenAI `text-embedding-3-small` (VECTOR_SIZE=1536).

```bash
cd duongspa-ai-sale-train-pipeline

# Chay auto test voi 7 cau hoi mau
python test/test_search_openai.py
```

**Yêu cầu:** `OPENAI_API_KEY` hợp lệ trong `.env`, Qdrant đang chạy, đã embed dữ liệu với `VECTOR_SIZE=1536`.

---

## test_search_local.py

Test toàn bộ pipeline search với local model `paraphrase-multilingual-MiniLM-L12-v2` (VECTOR_SIZE=384). **Không cần API key.**

```bash
cd duongspa-ai-sale-train-pipeline

# Cai dat truoc neu chua co
pip install sentence-transformers torch

# Chay auto test voi 7 cau hoi mau
python test/test_search_local.py

# Che do nhap tay — go cau hoi tuy y
python test/test_search_local.py --interactive
```

**Yêu cầu:** Qdrant đang chạy, đã embed dữ liệu với `EMBED_PROVIDER=local` và `VECTOR_SIZE=384`.

> ⚠️ **Lưu ý:** Dữ liệu trong Qdrant phải được embed bằng **cùng provider** khi chạy `main.py`. Không thể mix OpenAI vector (1536 dims) với local model (384 dims).

## Cách dùng

### Linux / macOS / WSL
```bash
chmod +x test_qdrant.sh
./test_qdrant.sh

# Override URL / collection / API key
QDRANT_URL=http://myserver:6333 \
COLLECTION=conversation_memory \
QDRANT_API_KEY=your-secret \
./test_qdrant.sh
```

### Windows PowerShell
```powershell
.\test_qdrant.ps1

# Override tham so
.\test_qdrant.ps1 -QdrantUrl "http://myserver:6333" `
                  -Collection "conversation_memory" `
                  -ApiKey "your-secret"
```

## Các test được thực hiện

| # | Test | Mô tả |
|---|---|---|
| 1 | Health check | Qdrant đang chạy không |
| 2 | List collections | Danh sách tất cả collections |
| 3 | Collection info | Vector size, distance, status, số points |
| 4 | Count points | Tổng số điểm đã upsert |
| 5 | Scroll 3 points | Xem payload mẫu |
| 6 | Search zero vector | Kiểm tra HNSW index hoạt động |
| 7 | Filter segment HIGH/VIP | Kiểm tra keyword index `segment` |
| 8 | Filter outcome_label=won | Kiểm tra keyword index `outcome_label` |
| 9 | Filter score > 80 | Kiểm tra float range index `score` |
| 10 | Filter has_closing_cta=true | Kiểm tra bool index |
| 11 | Payload indexes | Liệt kê các field đã được index |

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant server |
| `COLLECTION` | `conversation_memory` | Tên collection |
| `QDRANT_API_KEY` | _(trống)_ | API key nếu Qdrant bật auth |
