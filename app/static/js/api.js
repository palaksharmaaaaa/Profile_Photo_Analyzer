// Talking to the server: start a job, follow its stages, collect the result.

export const JOBS_URL = "/v1/jobs";
const POLL_MS = 700;
const FINISHED = new Set(["succeeded", "failed"]);

/** An error answer from the API (or a failure to reach it). `code` is the API's machine-readable reason. */
export class ApiProblem extends Error {
    constructor(message, { code = "unknown", status = 0, requestId = null, retryAfter = null } = {}) {
        super(message);
        this.name = "ApiProblem";
        this.code = code;
        this.status = status;
        this.requestId = requestId;
        this.retryAfter = retryAfter;
    }
}

/** Turn a failed response into an ApiProblem, using the API's error body when there is one. */
export async function problemFrom(response) {
    let body = {};
    try {
        body = await response.json();
    } catch {
        // not JSON: fall back to the status line below
    }
    return new ApiProblem(body.reason ?? `The server answered with status ${response.status}.`, {
        code: body.code,
        status: response.status,
        requestId: body.request_id ?? response.headers.get("X-Request-ID"),
        retryAfter: response.headers.get("Retry-After"),
    });
}

async function expectOk(response) {
    if (!response.ok) {
        throw await problemFrom(response);
    }
    return response.json();
}

/** Upload the photo and start an analysis. Resolves with {job_id, status, status_url, events_url}. */
export async function createJob(file, { fetchImpl = fetch, annotated = true, signal } = {}) {
    const body = new FormData();
    body.append("file", file);
    const response = await fetchImpl(`${JOBS_URL}?annotated_image=${annotated}`, { method: "POST", body, signal });
    return expectOk(response);
}

export async function getJob(statusUrl, { fetchImpl = fetch, signal } = {}) {
    return expectOk(await fetchImpl(statusUrl, { signal }));
}

const sleep = (ms, signal) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        signal?.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(new DOMException("Aborted", "AbortError"));
        });
    });

/** Ask for the job until it has finished, reporting every stage once. */
export async function pollJob(job, onStage, { fetchImpl = fetch, pollMs = POLL_MS, signal, seen = 0 } = {}) {
    const status = await getJob(job.status_url, { fetchImpl, signal });
    for (const stage of status.stages.slice(seen)) {
        onStage(stage);
    }
    if (FINISHED.has(status.status)) {
        return status;
    }
    await sleep(pollMs, signal);
    return pollJob(job, onStage, { fetchImpl, pollMs, signal, seen: status.stages.length });
}

/** Follow the server-sent events of a job; resolves with the number of stages seen once the server says "done". */
export function streamJob(job, onStage, { EventSourceImpl = globalThis.EventSource, signal } = {}) {
    return new Promise((resolve, reject) => {
        const source = new EventSourceImpl(job.events_url);
        let seen = 0;
        const close = () => source.close();
        signal?.addEventListener("abort", () => {
            close();
            reject(new DOMException("Aborted", "AbortError"));
        });
        source.addEventListener("stage", (event) => {
            seen += 1;
            onStage(JSON.parse(event.data));
        });
        source.addEventListener("done", () => {
            close();
            resolve(seen);
        });
        source.onerror = () => {
            close();
            reject(new ApiProblem("The progress stream was interrupted.", { code: "stream_interrupted" }));
        };
    });
}

/**
 * Run one analysis end to end: start it, show its stages as they happen (server-sent events, polling when those are
 * unavailable or break) and return the finished AnalysisResponse. Throws an ApiProblem when it fails.
 */
export async function analyse(file, onStage, options = {}) {
    const job = await createJob(file, options);
    let seen = 0;
    const count = (stage) => {
        seen += 1;
        onStage(stage);
    };
    let status;
    if (options.EventSourceImpl !== null && (options.EventSourceImpl ?? globalThis.EventSource)) {
        try {
            await streamJob(job, count, options);
            status = await getJob(job.status_url, options);
        } catch (error) {
            if (error.name === "AbortError") {
                throw error;
            }
        }
    }
    status ??= await pollJob(job, onStage, { ...options, seen });
    if (status.status === "failed") {
        throw new ApiProblem(status.error.reason, { code: status.error.code, requestId: status.error.request_id });
    }
    return status.result;
}
