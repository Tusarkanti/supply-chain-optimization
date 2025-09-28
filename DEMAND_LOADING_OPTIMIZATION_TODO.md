# Demand Forecasting Loading Optimization TODO

## Current Tasks
- [x] Add in-memory caching to /api/demand-forecast route in app.py (cache per product_id, TTL 1 hour)
- [x] Update getDemandForecast in api.js if needed (no change, as caching is backend)
- [x] Test: Generate forecast for a product, then re-generate immediately to verify cache hit (faster response)
- [x] Verify no impact on core ML logic or UI

## Followup
- [x] Restart backend after edits
- [x] Browser test: Select product, generate forecast (first slow, subsequent fast)

Optimization complete: Subsequent forecasts for the same product/user/horizon/country now use cache, reducing load time significantly.
