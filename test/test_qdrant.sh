#!/usr/bin/env bash
# =============================================================================
# test_qdrant.sh — Kiểm tra Qdrant sau khi pipeline embedding hoàn tất
# =============================================================================
# Cach dung:
#   chmod +x test_qdrant.sh
#   ./test_qdrant.sh
#
# Bien moi truong co the override:
#   QDRANT_URL=http://localhost:6333 COLLECTION=conversation_memory ./test_qdrant.sh
# =============================================================================

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${COLLECTION:-conversation_memory}"
API_KEY="${QDRANT_API_KEY:-}"          # De trong neu Qdrant chua bat API key

# Header auth (chi them neu co API key)
AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="-H \"api-key: ${API_KEY}\""
fi

# Helper: in tieu de
section() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# Helper: curl voi auth
_curl() {
  if [ -n "$API_KEY" ]; then
    curl -s -H "api-key: ${API_KEY}" "$@"
  else
    curl -s "$@"
  fi
}

# =============================================================================
# 1. Health check
# =============================================================================
section "1. Health check — ${QDRANT_URL}"
_curl "${QDRANT_URL}/healthz" | python3 -m json.tool 2>/dev/null || \
_curl "${QDRANT_URL}/healthz"

# =============================================================================
# 2. Danh sach tat ca collections
# =============================================================================
section "2. List collections"
_curl "${QDRANT_URL}/collections" | python3 -m json.tool 2>/dev/null || \
_curl "${QDRANT_URL}/collections"

# =============================================================================
# 3. Thong tin collection chinh
# =============================================================================
section "3. Collection info — '${COLLECTION}'"
_curl "${QDRANT_URL}/collections/${COLLECTION}" | python3 -m json.tool 2>/dev/null || \
_curl "${QDRANT_URL}/collections/${COLLECTION}"

# =============================================================================
# 4. Dem tong so points
# =============================================================================
section "4. Count points"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/count" \
  -H "Content-Type: application/json" \
  -d '{"exact": true}' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 5. Lay 3 points dau tien (scroll)
# =============================================================================
section "5. Scroll — lay 3 points dau tien (co payload, khong lay vector)"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 3,
    "with_payload": true,
    "with_vector": false
  }' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 6. Search theo vector gia (zero vector) — kiem tra index hoat dong
# =============================================================================
section "6. Search voi zero vector (top 3) — kiem tra index"
# Lay vector_size tu collection info
VECTOR_SIZE=$(_curl "${QDRANT_URL}/collections/${COLLECTION}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['config']['params']['vectors']['size'])" 2>/dev/null || echo "1536")
echo "   Vector size: ${VECTOR_SIZE}"

ZERO_VEC=$(python3 -c "print('[' + ','.join(['0.0']*${VECTOR_SIZE}) + ']')")

_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"vector\": ${ZERO_VEC},
    \"limit\": 3,
    \"with_payload\": true,
    \"with_vector\": false
  }" | python3 -m json.tool 2>/dev/null

# =============================================================================
# 7. Filter theo segment = HIGH hoac VIP
# =============================================================================
section "7. Filter points — segment IN [HIGH, VIP] (top 5)"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "should": [
        {"key": "segment", "match": {"value": "HIGH"}},
        {"key": "segment", "match": {"value": "VIP"}}
      ]
    }
  }' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 8. Filter theo outcome_label = won
# =============================================================================
section "8. Filter points — outcome_label = won (top 5)"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [
        {"key": "outcome_label", "match": {"value": "won"}}
      ]
    }
  }' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 9. Filter theo score > 80
# =============================================================================
section "9. Filter points — score > 80 (top 5)"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [
        {"key": "score", "range": {"gt": 80}}
      ]
    }
  }' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 10. Filter theo has_closing_cta = true
# =============================================================================
section "10. Filter points — has_closing_cta = true (top 5)"
_curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 5,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [
        {"key": "has_closing_cta", "match": {"value": true}}
      ]
    }
  }' | python3 -m json.tool 2>/dev/null

# =============================================================================
# 11. Lay payload indexes
# =============================================================================
section "11. Payload indexes cua collection"
_curl "${QDRANT_URL}/collections/${COLLECTION}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
indexes = d.get('result', {}).get('payload_schema', {})
if indexes:
    for k, v in indexes.items():
        print(f'  {k}: {v.get(\"data_type\", v)}')
else:
    print('  (khong co index hoac chua duoc tao)')
" 2>/dev/null

echo
echo "✅ Done. Qdrant URL: ${QDRANT_URL} | Collection: ${COLLECTION}"
