# =============================================================================
# test_qdrant.ps1 — Kiểm tra Qdrant sau khi pipeline embedding hoàn tất
# =============================================================================
# Cach dung (PowerShell):
#   .\test_qdrant.ps1
#   .\test_qdrant.ps1 -QdrantUrl "http://myserver:6333" -Collection "my_col" -ApiKey "secret"
# =============================================================================

param(
    [string]$QdrantUrl  = $env:QDRANT_URL       ?? "http://localhost:6333",
    [string]$Collection = $env:COLLECTION        ?? "conversation_memory",
    [string]$ApiKey     = $env:QDRANT_API_KEY    ?? ""
)

$Headers = @{ "Content-Type" = "application/json" }
if ($ApiKey) { $Headers["api-key"] = $ApiKey }

function Section($title) {
    Write-Host ""
    Write-Host ("━" * 55) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("━" * 55) -ForegroundColor Cyan
}

function Invoke-Qdrant($Method, $Path, $Body = $null) {
    $uri = "$QdrantUrl$Path"
    try {
        if ($Body) {
            $resp = Invoke-RestMethod -Method $Method -Uri $uri -Headers $Headers `
                        -Body ($Body | ConvertTo-Json -Depth 10) -ErrorAction Stop
        } else {
            $resp = Invoke-RestMethod -Method $Method -Uri $uri -Headers $Headers -ErrorAction Stop
        }
        return $resp
    } catch {
        Write-Host "  [ERROR] $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# =============================================================================
# 1. Health check
# =============================================================================
Section "1. Health check — $QdrantUrl"
$r = Invoke-Qdrant GET "/healthz"
$r | ConvertTo-Json -Depth 5

# =============================================================================
# 2. Danh sach tat ca collections
# =============================================================================
Section "2. List collections"
$r = Invoke-Qdrant GET "/collections"
$r.result.collections | ForEach-Object { Write-Host "  - $($_.name)" }

# =============================================================================
# 3. Thong tin collection chinh
# =============================================================================
Section "3. Collection info — '$Collection'"
$r = Invoke-Qdrant GET "/collections/$Collection"
if ($r) {
    $cfg = $r.result.config.params
    Write-Host "  vectors size : $($cfg.vectors.size)"
    Write-Host "  distance     : $($cfg.vectors.distance)"
    Write-Host "  points count : $($r.result.points_count)"
    Write-Host "  status       : $($r.result.status)"
}

# =============================================================================
# 4. Dem tong so points
# =============================================================================
Section "4. Count points (exact)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/count" @{ exact = $true }
Write-Host "  Total points: $($r.result.count)"

# =============================================================================
# 5. Scroll — lay 3 points dau tien
# =============================================================================
Section "5. Scroll — 3 points dau tien (payload only)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/scroll" @{
    limit        = 3
    with_payload = $true
    with_vector  = $false
}
$r.result.points | ForEach-Object {
    Write-Host "  [id=$($_.id)] segment=$($_.payload.segment) | outcome=$($_.payload.outcome_label) | score=$($_.payload.score)"
}

# =============================================================================
# 6. Search voi zero vector
# =============================================================================
Section "6. Search voi zero vector (top 3) — kiem tra index"
$vectorSize = (Invoke-Qdrant GET "/collections/$Collection").result.config.params.vectors.size
if (-not $vectorSize) { $vectorSize = 1536 }
Write-Host "  Vector size: $vectorSize"
$zeroVec = @(0.0) * $vectorSize

$r = Invoke-Qdrant POST "/collections/$Collection/points/search" @{
    vector       = $zeroVec
    limit        = 3
    with_payload = $true
    with_vector  = $false
}
$r.result | ForEach-Object {
    Write-Host "  [id=$($_.id)] score=$($_.score) | segment=$($_.payload.segment) | outcome=$($_.payload.outcome_label)"
}

# =============================================================================
# 7. Filter: segment IN [HIGH, VIP]
# =============================================================================
Section "7. Filter — segment IN [HIGH, VIP] (top 5)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/scroll" @{
    limit        = 5
    with_payload = $true
    with_vector  = $false
    filter       = @{
        should = @(
            @{ key = "segment"; match = @{ value = "HIGH" } },
            @{ key = "segment"; match = @{ value = "VIP"  } }
        )
    }
}
$r.result.points | ForEach-Object {
    Write-Host "  [id=$($_.id)] segment=$($_.payload.segment) | score=$($_.payload.score) | outcome=$($_.payload.outcome_label)"
}

# =============================================================================
# 8. Filter: outcome_label = won
# =============================================================================
Section "8. Filter — outcome_label = won (top 5)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/scroll" @{
    limit        = 5
    with_payload = $true
    with_vector  = $false
    filter       = @{
        must = @(
            @{ key = "outcome_label"; match = @{ value = "won" } }
        )
    }
}
$r.result.points | ForEach-Object {
    Write-Host "  [id=$($_.id)] segment=$($_.payload.segment) | weighted_score=$($_.payload.weighted_score)"
}

# =============================================================================
# 9. Filter: score > 80
# =============================================================================
Section "9. Filter — score > 80 (top 5)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/scroll" @{
    limit        = 5
    with_payload = $true
    with_vector  = $false
    filter       = @{
        must = @(
            @{ key = "score"; range = @{ gt = 80 } }
        )
    }
}
$r.result.points | ForEach-Object {
    Write-Host "  [id=$($_.id)] score=$($_.payload.score) | segment=$($_.payload.segment) | outcome=$($_.payload.outcome_label)"
}

# =============================================================================
# 10. Filter: has_closing_cta = true
# =============================================================================
Section "10. Filter — has_closing_cta = true (top 5)"
$r = Invoke-Qdrant POST "/collections/$Collection/points/scroll" @{
    limit        = 5
    with_payload = $true
    with_vector  = $false
    filter       = @{
        must = @(
            @{ key = "has_closing_cta"; match = @{ value = $true } }
        )
    }
}
$r.result.points | ForEach-Object {
    Write-Host "  [id=$($_.id)] has_closing_cta=$($_.payload.has_closing_cta) | segment=$($_.payload.segment)"
}

# =============================================================================
# 11. Payload indexes
# =============================================================================
Section "11. Payload indexes"
$r = Invoke-Qdrant GET "/collections/$Collection"
$schema = $r.result.payload_schema
if ($schema) {
    $schema.PSObject.Properties | ForEach-Object {
        Write-Host "  $($_.Name): $($_.Value.data_type)"
    }
} else {
    Write-Host "  (khong co payload_schema)"
}

Write-Host ""
Write-Host "✅ Done. URL: $QdrantUrl | Collection: $Collection" -ForegroundColor Green
