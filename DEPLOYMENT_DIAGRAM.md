# Supply Chain Optimization - Local vs Production Setup Diagram

## Overview
This diagram shows how the React frontend connects to the Flask backend in different environments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENVIRONMENT SETUP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                    ┌─────────────────┐                 │
│  │   LOCAL DEV     │                    │   PRODUCTION     │                 │
│  │                 │                    │                 │                 │
│  │  Frontend:      │                    │  Frontend:      │                 │
│  │  npm start      │                    │  Render Deploy  │                 │
│  │  (localhost:3000)│                    │  (Render URL)   │                 │
│  │                 │                    │                 │                 │
│  │  Backend:       │                    │  Backend:       │                 │
│  │  python app.py  │                    │  Render Deploy  │                 │
│  │  (localhost:5000)│                    │  (supply-chain-│                 │
│  │                 │                    │     optimization-│                 │
│  │                 │                    │     2.onrender.com)│                 │
│  └─────────────────┘                    └─────────────────┘                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          API URL CONFIGURATION                          │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  Frontend reads REACT_APP_API_URL from:                                │ │
│  │                                                                         │ │
│  │  LOCAL:     frontend/.env file                                         │ │
│  │           ↳ REACT_APP_API_URL=http://localhost:5000                    │ │
│  │                                                                         │ │
│  │  PRODUCTION: Render environment variables                              │ │
│  │           ↳ REACT_APP_API_URL=https://supply-chain-optimization-2.onrender.com │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          REQUEST FLOW                                   │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  User Action → Frontend Component → api.js → API Call                  │ │
│  │                                                                         │ │
│  │  Example: Login Form                                                    │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  │   Login.js  │───▶│   api.js    │───▶│  process.env │───▶│  Backend   │ │
│  │  │             │    │             │    │ REACT_APP_  │    │             │ │
│  │  │ User enters │    │ fetch(`${API}│    │ API_URL     │    │ /api/login │ │
│  │  │ credentials │    │ /api/login`)│    │             │    │             │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│  │                                                                         │ │
│  │  LOCAL:      http://localhost:5000/api/login                           │ │
│  │  PRODUCTION: https://supply-chain-optimization-2.onrender.com/api/login │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          CORS CONFIGURATION                             │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  Backend (ml-service/app.py) allows origins:                           │ │
│  │                                                                         │ │
│  │  ✅ https://supply-chain-frontend.onrender.com (default Render URL)    │ │
│  │  ✅ https://supply-chain-optimization-v93t.onrender.com (custom domain)│ │
│  │  ✅ http://localhost:3000 (local development)                          │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Points

### Local Development
1. Start backend: `python ml-service/app.py` (runs on localhost:5000)
2. Start frontend: `cd frontend && npm start` (runs on localhost:3000)
3. Frontend reads `REACT_APP_API_URL` from `frontend/.env`
4. API calls go to `http://localhost:5000`

### Production Deployment
1. Backend deployed to Render at `https://supply-chain-optimization-2.onrender.com`
2. Frontend deployed to Render with `REACT_APP_API_URL` env var set
3. API calls go to production backend URL
4. CORS allows the frontend origin

### Files Modified
- `frontend/.env` - Local API URL configuration
- `ml-service/app.py` - CORS origins updated
- `render.yaml` - Production environment variables (already configured)

This setup ensures seamless development and deployment without hardcoded URLs.
