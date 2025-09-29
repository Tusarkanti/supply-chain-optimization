# Supply Chain Optimization - API URL Configuration Fix

## Completed Tasks

### 1. Create .env file for local development
- **Status**: ✅ Completed
- **Description**: Created `frontend/.env` with `REACT_APP_API_URL=http://localhost:5000` for local development
- **Files Modified**: `frontend/.env`

### 2. Update CORS origins in backend
- **Status**: ✅ Completed
- **Description**: Updated CORS in `ml-service/app.py` to allow both default Render URLs and custom domain
- **Files Modified**: `ml-service/app.py`
- **Changes**: Added "https://supply-chain-frontend.onrender.com" to allowed origins

### 3. Verify production configuration
- **Status**: ✅ Verified
- **Description**: Confirmed `render.yaml` has correct `REACT_APP_API_URL=https://supply-chain-optimization-2.onrender.com` for production

## Summary

The API URL configuration issue has been resolved:

- **Local Development**: Frontend now uses `http://localhost:5000` via `.env` file
- **Production**: Frontend uses `https://supply-chain-optimization-2.onrender.com` via Render environment variable
- **CORS**: Backend allows requests from both default Render frontend URL and any custom domains

The application should now work correctly in both local and production environments.
