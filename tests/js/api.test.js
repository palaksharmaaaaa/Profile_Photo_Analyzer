import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { analyse, ApiProblem, createJob, getJob, pollJob, problemFrom, streamJob } from "../../app/static/js/api.js";
import { FakeEventSource, fixture, jsonResponse } from "./helpers.js";

const JOB = { job_id: "j1", status: "queued", status_url: "/v1/jobs/j1", events_url: "/v1/jobs/j1/events" };
const stage = (n, status = "completed") => ({ step_id: `s${n}`, step_number: n, title: `Step ${n}`, status, detail: "", elapsed_ms: 1 });
const photo = new File(["x"], "p.jpg", { type: "image/jpeg" });
const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

/** A fetch that answers the job endpoints from a script of statuses. */
function jobFetch(statuses, { create = () => jsonResponse(JOB, { status: 202 }) } = {}) {
    const calls = [];
    let index = 0;
    const impl = async (url, options) => {
        calls.push({ url, options });
        if (options?.method === "POST") {
            return create();
        }
        const next = statuses[Math.min(index, statuses.length - 1)];
        index += 1;
        return next instanceof Response ? next : jsonResponse(next);
    };
    impl.calls = calls;
    return impl;
}

const job = (status, stages = [], extra = {}) => ({ job_id: "j1", status, stages, result: null, error: null, ...extra });

describe("problems", () => {
    it("carry the code, status, request id and retry hint of the API answer", async () => {
        const response = jsonResponse(
            { code: "rate_limited", reason: "Too many requests.", request_id: "r-1" },
            { status: 429, headers: { "Retry-After": "60" } },
        );
        const problem = await problemFrom(response);
        assert.ok(problem instanceof ApiProblem);
        assert.deepEqual(
            [problem.message, problem.code, problem.status, problem.requestId, problem.retryAfter],
            ["Too many requests.", "rate_limited", 429, "r-1", "60"],
        );
    });

    it("fall back to the status and the request id header when the body is not JSON", async () => {
        const problem = await problemFrom(new Response("<html>", { status: 502, headers: { "X-Request-ID": "h-2" } }));
        assert.equal(problem.message, "The server answered with status 502.");
        assert.equal(problem.code, "unknown");
        assert.equal(problem.requestId, "h-2");
        assert.equal(problem.retryAfter, null);
        assert.equal(new ApiProblem("x").status, 0);
    });
});

describe("createJob and getJob", () => {
    it("post the photo and ask for the annotated image by default", async () => {
        const fetchImpl = jobFetch([]);
        assert.deepEqual(await createJob(photo, { fetchImpl }), JOB);
        const { url, options } = fetchImpl.calls[0];
        assert.equal(url, "/v1/jobs?annotated_image=true");
        assert.equal(options.method, "POST");
        assert.equal(options.body.get("file").name, "p.jpg");
        await createJob(photo, { fetchImpl, annotated: false });
        assert.equal(fetchImpl.calls[1].url, "/v1/jobs?annotated_image=false");
    });

    it("throw the error the API reports", async () => {
        const create = () => jsonResponse({ code: "invalid_image", reason: "Not an image." }, { status: 400 });
        await assert.rejects(createJob(photo, { fetchImpl: jobFetch([], { create }) }), (error) => error.code === "invalid_image");
        const gone = async () => jsonResponse({ code: "job_not_found", reason: "Gone" }, { status: 404 });
        await assert.rejects(getJob("/v1/jobs/x", { fetchImpl: gone }), /Gone/);
    });
});

describe("pollJob", () => {
    it("reports each stage once and returns the finished job", async () => {
        const statuses = [job("running", [stage(1)]), job("running", [stage(1), stage(2)]), job("succeeded", [stage(1), stage(2), stage(3)])];
        const seen = [];
        const result = await pollJob(JOB, (s) => seen.push(s.step_number), { fetchImpl: jobFetch(statuses), pollMs: 1 });
        assert.equal(result.status, "succeeded");
        assert.deepEqual(seen, [1, 2, 3]);
    });

    it("skips the stages that were already reported", async () => {
        const seen = [];
        await pollJob(JOB, (s) => seen.push(s.step_number), { fetchImpl: jobFetch([job("failed", [stage(1), stage(2)])]), seen: 1 });
        assert.deepEqual(seen, [2]);
    });

    it("stops waiting when it is aborted", async () => {
        const controller = new AbortController();
        const pending = pollJob(JOB, () => {}, { fetchImpl: jobFetch([job("running")]), pollMs: 1000, signal: controller.signal });
        setTimeout(() => controller.abort(), 5);
        await assert.rejects(pending, (error) => error.name === "AbortError");
    });
});

describe("streamJob", () => {
    it("passes every stage on and resolves when the server says done", async () => {
        const seen = [];
        const pending = streamJob(JOB, (s) => seen.push(s.step_number), { EventSourceImpl: FakeEventSource });
        const source = FakeEventSource.last;
        assert.equal(source.url, JOB.events_url);
        source.emit("stage", stage(1));
        source.emit("stage", stage(2));
        source.listeners.done({});
        assert.equal(await pending, 2);
        assert.deepEqual(seen, [1, 2]);
        assert.ok(source.closed);
    });

    it("rejects when the stream breaks", async () => {
        const pending = streamJob(JOB, () => {}, { EventSourceImpl: FakeEventSource });
        FakeEventSource.last.onerror();
        await assert.rejects(pending, (error) => error.code === "stream_interrupted");
        assert.ok(FakeEventSource.last.closed);
    });

    it("closes the stream when it is aborted", async () => {
        const controller = new AbortController();
        const pending = streamJob(JOB, () => {}, { EventSourceImpl: FakeEventSource, signal: controller.signal });
        controller.abort();
        await assert.rejects(pending, (error) => error.name === "AbortError");
        assert.ok(FakeEventSource.last.closed);
    });
});

describe("analyse", () => {
    const result = fixture("result.json");

    it("follows the event stream and returns the result", async () => {
        const fetchImpl = jobFetch([job("succeeded", [stage(1), stage(2)], { result })]);
        const seen = [];
        const pending = analyse(photo, (s) => seen.push(s.step_number), { fetchImpl, EventSourceImpl: FakeEventSource });
        await tick();
        FakeEventSource.last.emit("stage", stage(1));
        FakeEventSource.last.emit("stage", stage(2));
        FakeEventSource.last.listeners.done({});
        assert.deepEqual(await pending, result);
        assert.deepEqual(seen, [1, 2]);
    });

    it("falls back to polling, without repeating stages, when the stream breaks", async () => {
        const fetchImpl = jobFetch([job("succeeded", [stage(1), stage(2), stage(3)], { result })]);
        const seen = [];
        const pending = analyse(photo, (s) => seen.push(s.step_number), { fetchImpl, EventSourceImpl: FakeEventSource, pollMs: 1 });
        await tick();
        FakeEventSource.last.emit("stage", stage(1));
        FakeEventSource.last.onerror();
        assert.deepEqual(await pending, result);
        assert.deepEqual(seen, [1, 2, 3]);
    });

    it("polls when the browser has no event streams", async () => {
        const fetchImpl = jobFetch([job("running", [stage(1)]), job("succeeded", [stage(1), stage(2)], { result })]);
        const seen = [];
        assert.deepEqual(await analyse(photo, (s) => seen.push(s.step_number), { fetchImpl, EventSourceImpl: null, pollMs: 1 }), result);
        assert.deepEqual(seen, [1, 2]);
    });

    it("uses the global EventSource when there is one", async () => {
        globalThis.EventSource = FakeEventSource;
        try {
            const pending = analyse(photo, () => {}, { fetchImpl: jobFetch([job("succeeded", [stage(1)], { result })]) });
            await tick();
            FakeEventSource.last.listeners.done({});
            assert.deepEqual(await pending, result);
        } finally {
            delete globalThis.EventSource;
        }
    });

    it("polls when nothing provides event streams at all", async () => {
        const fetchImpl = jobFetch([job("succeeded", [], { result })]);
        assert.deepEqual(await analyse(photo, () => {}, { fetchImpl, pollMs: 1 }), result);
    });

    it("turns a failed job into an ApiProblem", async () => {
        const failed = job("failed", [stage(1)], { error: { code: "analysis_failed", reason: "Analysis failed.", request_id: "r-9" } });
        await assert.rejects(
            analyse(photo, () => {}, { fetchImpl: jobFetch([failed]), EventSourceImpl: null }),
            (error) => error.code === "analysis_failed" && error.requestId === "r-9" && error.message === "Analysis failed.",
        );
    });

    it("passes an abort through instead of falling back", async () => {
        const controller = new AbortController();
        const options = { fetchImpl: jobFetch([job("running")]), EventSourceImpl: FakeEventSource, signal: controller.signal };
        const pending = analyse(photo, () => {}, options);
        await tick();
        controller.abort();
        await assert.rejects(pending, (error) => error.name === "AbortError");
    });

    it("does not start when the upload is refused", async () => {
        const create = () => jsonResponse({ code: "queue_full", reason: "Busy." }, { status: 503, headers: { "Retry-After": "5" } });
        await assert.rejects(analyse(photo, () => {}, { fetchImpl: jobFetch([], { create }) }), (error) => error.retryAfter === "5");
    });
});
