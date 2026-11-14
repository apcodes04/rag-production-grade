/**
 * One typed place for every call to the backend.
 *
 * Why a wrapper and not fetch() scattered through components: the API key
 * header, the base URL and the error shape are decided once. A component that
 * does its own fetch is a component that forgets one of the three.
 */
import type {
  AnswerResponse, DocumentsResponse, IndexStats, RunDetail, RunSummary, UploadResult,
} from "./types";

export const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const KEY = import.meta.env.VITE_API_KEY ?? "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly requestId?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        // FormData must set its own Content-Type: the browser appends the
        // multipart boundary, and overriding it makes the body unparseable.
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(KEY ? { "X-API-Key": KEY } : {}),
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    // fetch() rejects without a status for several different causes, and the
    // browser deliberately hides which one: the server being down, and a CORS
    // preflight being refused, look identical here. Saying only "is the server
    // running" sent us hunting a live server once already.
    throw new ApiError(
      `Cannot reach the API at ${BASE}. Either it is not running, or it refused ` +
      `this request's CORS preflight (check that ${init?.method ?? "GET"} is in ` +
      `allow_methods and that this origin is in CORS_ORIGINS).`, 0);
  }
  const requestId = response.headers.get("X-Request-ID") ?? undefined;
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? body.error ?? response.statusText, response.status, requestId);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<IndexStats>("/health"),
  runs: () => request<{ runs: RunSummary[] }>("/runs").then((r) => r.runs),
  run: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  ask: (question: string, useCache = true, sources: string[] = []) =>
    request<AnswerResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, history: [], use_cache: useCache, sources }),
    }),
  /** Called when the tab closes, so an interviewer's upload does not outlive
   *  their visit. sendBeacon survives the page going away, which a normal
   *  fetch does not: the browser cancels in flight requests on unload. */
  cleanupUploadsOnLeave: () => {
    try {
      navigator.sendBeacon(`${BASE}/documents/cleanup`);
    } catch {
      /* nothing useful to do while the page is being torn down */
    }
  },
  documents: () => request<DocumentsResponse>("/documents"),
  // A plain URL, not a fetch: the browser downloads it directly, which keeps
  // the file out of JS memory and gives the user a real save dialog.
  downloadUrl: (path: string) => `${BASE}/documents/${encodeURIComponent(path)}/download`,
  // Same file, but the backend marks it inline so a new tab renders it
  // (a PDF opens in the browser's own viewer) instead of saving it.
  viewUrl: (path: string) => `${BASE}/documents/${encodeURIComponent(path)}/view`,
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadResult>("/documents", { method: "POST", body });
  },
  removeDocument: (path: string) =>
    request<{ removed: string; chunks_removed: number }>(
      `/documents/${encodeURIComponent(path)}`, { method: "DELETE" }),
};
