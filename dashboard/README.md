# RAG Observability Dashboard

React + Vite + TypeScript + Tailwind + Recharts. Reads the FastAPI backend in
`../src/rag/api.py`: evaluation runs, per question drill down, and a live query
box that exercises the same pipeline the metrics describe.

```bash
npm install
cp .env.example .env.local        # point VITE_API_URL at the API
npm run dev                       # http://localhost:5173
```

The API must be running:

```bash
cd ..
uvicorn rag.api:app --app-dir src --reload
```

Build for deployment: `npm run build`, output in `dist/`.
