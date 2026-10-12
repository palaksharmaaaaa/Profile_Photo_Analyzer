import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";
import { ApiProblem } from "../../app/static/js/api.js";
import { describeProblem, init, LIMITS_URL } from "../../app/static/js/app.js";
import { $ } from "../../app/static/js/dom.js";
import { fixture, giveSize, jsonResponse, loadPage } from "./helpers.js";

const photo = (name = "me.jpg", size = 2 * 1024 * 1024) => Object.assign(new File(["x"], name, { type: "image/jpeg" }), { fakeSize: size });
const sized = (file, size) => Object.defineProperty(file, "size", { value: size });

class FakeObserver {
    static last = null;

    constructor(callback) {
        this.callback = callback;
        this.observed = [];
        FakeObserver.last = this;
    }

    observe(node) {
        this.observed.push(node);
    }
}

/** Fake browser pieces; `calls` records what the page asked of them. */
function fakes(overrides = {}) {
    const calls = { created: [], revoked: [], analysed: [], fitted: [], limitRequests: 0 };
    const deps = {
        analyseImpl: async (file, onStage, options) => {
            calls.analysed.push({ file, options });
            [1, 2, 3, 4].forEach((n) => onStage({ step_number: n, status: "completed" }));
            return structuredClone(fixture("result.json"));
        },
        fitImpl: async (file, maxBytes) => {
            calls.fitted.push([file, maxBytes]);
            return { file, shrunk: false };
        },
        fetchImpl: async (url) => {
            assert.equal(url, LIMITS_URL);
            calls.limitRequests += 1;
            return jsonResponse({ max_upload_bytes: 10 * 1024 * 1024 });
        },
        createObjectUrl: (blob) => {
            calls.created.push(blob);
            return `blob:${calls.created.length}`;
        },
        revokeObjectUrl: (url) => calls.revoked.push(url),
        ResizeObserverImpl: FakeObserver,
        scrollOptions: () => ({ behavior: "auto" }),
        ...overrides,
    };
    return { deps, calls };
}

function choose(file) {
    Object.defineProperty($("photo"), "files", { value: [file], configurable: true });
    $("photo").dispatchEvent(new window.Event("change"));
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));

describe("describeProblem", () => {
    it("explains an API problem with a wait and a reference, and an unreachable server plainly", () => {
        const problem = new ApiProblem("Busy.", { retryAfter: "5", requestId: "r-1" });
        assert.equal(describeProblem(problem), "Busy. Try again in 5 seconds. (reference r-1)");
        assert.equal(describeProblem(new ApiProblem("Bad.")), "Bad.");
        assert.match(describeProblem(new TypeError("fetch failed")), /could not be reached/);
    });
});

describe("the page", () => {
    beforeEach(() => loadPage());

    it("learns the limits of the server and says so", async () => {
        const { deps, calls } = fakes();
        const page = init(deps);
        await settle();
        assert.equal(page.state.maxBytes, 10 * 1024 * 1024);
        assert.match($("upload-hint").textContent, /up to 10.00 MB$/);
        assert.equal(calls.limitRequests, 1);
        assert.deepEqual(FakeObserver.last.observed, [$("original-image")]);
    });

    it("works without the limits when they cannot be fetched", async () => {
        for (const fetchImpl of [async () => jsonResponse({}, { status: 500 }), async () => Promise.reject(new Error("offline"))]) {
            loadPage();
            const page = init(fakes({ fetchImpl }).deps);
            await settle();
            assert.equal(page.state.maxBytes, null);
            assert.doesNotMatch($("upload-hint").textContent, /up to/);
        }
    });

    it("shows the chosen photo in the drop zone and remembers it", () => {
        const { deps, calls } = fakes();
        const page = init(deps);
        choose(sized(photo("a.jpg"), 3 * 1024 * 1024));
        assert.equal($("upload-title-text").textContent, "a.jpg");
        assert.match($("upload-hint").textContent, /^3.00 MB - click to choose/);
        assert.equal($("drop-preview").getAttribute("src"), "blob:1");
        assert.equal($("loading-preview").getAttribute("src"), "blob:1");
        assert.ok($("upload-visual").classList.contains("hidden"));
        choose(photo("b.jpg"));
        assert.deepEqual(calls.revoked, ["blob:1"]);
        assert.equal(page.state.photoUrl, "blob:2");
        $("photo").dispatchEvent(new window.Event("change")); // same file again: still fine
    });

    it("uses the browser's own object URLs when none are injected", () => {
        const { deps } = fakes();
        delete deps.createObjectUrl;
        delete deps.revokeObjectUrl;
        const page = init(deps);
        choose(photo("a.jpg"));
        assert.match(page.state.photoUrl, /^blob:/);
        const first = page.state.photoUrl;
        choose(photo("b.jpg"));
        assert.notEqual(page.state.photoUrl, first);
    });

    it("ignores a change event without a file", () => {
        init(fakes().deps);
        Object.defineProperty($("photo"), "files", { value: [], configurable: true });
        $("photo").dispatchEvent(new window.Event("change"));
        assert.ok(!$("upload-visual").classList.contains("hidden"));
    });

    it("accepts a dropped photo and highlights the zone while it is dragged over", () => {
        init(fakes().deps);
        const zone = $("drop-zone");
        zone.dispatchEvent(new window.Event("dragenter"));
        assert.ok(zone.classList.contains("dragover"));
        zone.dispatchEvent(new window.Event("dragleave"));
        assert.ok(!zone.classList.contains("dragover"));
        zone.dispatchEvent(new window.Event("dragover"));
        assert.ok(zone.classList.contains("dragover"));

        const drop = new window.Event("drop");
        drop.dataTransfer = { files: [] };
        zone.dispatchEvent(drop);
        assert.ok(!zone.classList.contains("dragover"));
        assert.equal($("upload-title-text").textContent.includes("Drop a photo"), true);

        const file = photo("dropped.jpg");
        const real = new window.Event("drop");
        real.dataTransfer = { files: [file] };
        Object.defineProperty($("photo"), "files", { set() {}, get: () => [file], configurable: true });
        zone.dispatchEvent(real);
        assert.equal($("upload-title-text").textContent, "dropped.jpg");
    });

    it("analyses the chosen photo, shows each step and the result, and scrolls to it", async () => {
        const { deps, calls } = fakes();
        const page = init(deps);
        await settle();
        choose(photo());
        const finished = page.analyseChosenPhoto({ preventDefault() {} });
        assert.ok($("analyze-button").disabled);
        await finished;
        assert.equal(calls.analysed.length, 1);
        assert.equal(calls.analysed[0].file.name, "me.jpg");
        assert.ok(calls.analysed[0].options.signal instanceof AbortSignal);
        assert.equal(calls.fitted.length, 1);
        assert.ok(!$("result-panel").classList.contains("hidden"));
        assert.ok(!$("reset-button").classList.contains("hidden"));
        assert.ok($("loading-panel").classList.contains("hidden"));
        assert.equal($("verdict").textContent, "A non-human face is also in the photo");
        assert.equal($("original-image").getAttribute("src"), "blob:1");
        assert.deepEqual($("result-panel").scrolledWith, { block: "start", behavior: "auto" });
        assert.equal(page.state.result.status, "NON_HUMAN_FACE_PRESENT");
    });

    it("does nothing when no photo was chosen", async () => {
        const { deps, calls } = fakes();
        const page = init(deps);
        Object.defineProperty($("photo"), "files", { value: [], configurable: true });
        await page.analyseChosenPhoto();
        assert.equal(calls.analysed.length, 0);
        assert.ok($("result-panel").classList.contains("hidden"));
    });

    it("submits through the form", async () => {
        const { deps, calls } = fakes();
        init(deps);
        choose(photo());
        $("upload-form").dispatchEvent(new window.Event("submit", { cancelable: true }));
        await settle();
        assert.equal(calls.analysed.length, 1);
    });

    it("sends a shrunk copy when the photo was over the limit", async () => {
        const small = new File(["y"], "me.jpg", { type: "image/jpeg" });
        const { deps, calls } = fakes({ fitImpl: async () => ({ file: small, shrunk: true }) });
        const page = init(deps);
        await settle();
        choose(photo());
        await page.analyseChosenPhoto();
        assert.equal(calls.analysed[0].file, small);
        assert.equal(page.state.photoUrl, "blob:2");
        assert.deepEqual(calls.revoked, ["blob:1"]);
    });

    it("sends the photo as it is when the limits are unknown", async () => {
        const { deps, calls } = fakes({ fetchImpl: async () => jsonResponse({}, { status: 500 }) });
        const page = init(deps);
        await settle();
        const file = photo();
        choose(file);
        await page.analyseChosenPhoto();
        assert.equal(calls.fitted.length, 0);
        assert.equal(calls.analysed[0].file, file);
    });

    it("explains a failure and marks the first step as failed", async () => {
        const failing = fakes({
            analyseImpl: async () => {
                throw new ApiProblem("Not an image.", { code: "invalid_image", requestId: "r-5" });
            },
        });
        const page = init(failing.deps);
        choose(photo());
        await page.analyseChosenPhoto();
        assert.equal($("error-text").textContent, "Not an image. (reference r-5)");
        assert.ok(!$("error-panel").classList.contains("hidden"));
        assert.equal(document.querySelector('.step[data-step="1"] .pill').textContent, "Failed");
        assert.ok(!$("analyze-button").disabled);
    });

    it("stays quiet when an analysis is cancelled by a newer one, whenever that happens", async () => {
        const signals = [];
        const slow = fakes({
            analyseImpl: (file, onStage, { signal }) =>
                new Promise((resolve, reject) => {
                    signals.push(signal);
                    signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
                    setTimeout(() => resolve(structuredClone(fixture("result.json"))), 20);
                }),
        });
        const page = init(slow.deps);
        choose(photo());

        // a newer analysis starts while the first is still preparing its photo: the first never reaches the server
        await Promise.all([page.analyseChosenPhoto(), page.analyseChosenPhoto()]);
        assert.equal(signals.length, 1);

        // a newer analysis starts while the first is already running: the first is cancelled
        const running = page.analyseChosenPhoto();
        await settle();
        const newer = page.analyseChosenPhoto();
        await Promise.all([running, newer]);
        assert.equal(signals.length, 3);
        assert.ok(signals[1].aborted);
        assert.ok(!signals[2].aborted);
        assert.ok($("error-panel").classList.contains("hidden"));
        assert.ok(!$("result-panel").classList.contains("hidden"));
        assert.ok($("loading-panel").classList.contains("hidden"));
        assert.ok(!$("analyze-button").disabled);
    });

    it("starts over, cancelling anything that is running", async () => {
        const { deps } = fakes();
        const page = init(deps);
        choose(photo());
        await page.analyseChosenPhoto();
        const controller = page.state.controller;
        $("analyze-another").click();
        assert.ok(controller.signal.aborted);
        assert.equal(page.state.result, null);
        for (const id of ["result-panel", "error-panel", "loading-panel", "reset-button"]) {
            assert.ok($(id).classList.contains("hidden"), id);
        }
        assert.ok(!$("upload-visual").classList.contains("hidden"));
        assert.ok($("drop-preview").classList.contains("hidden"));
        assert.equal($("upload-title-text").textContent, "Drop a photo here, or browse");
        assert.deepEqual(window.scrolledTo, { top: 0, behavior: "auto" });
        $("reset-button").click();
        init(fakes().deps).reset(); // a fresh page with nothing running resets too
    });

    it("exports the report without the image, and the annotated image as it is", async () => {
        const { deps, calls } = fakes();
        const page = init(deps);
        $("export-json").click();
        $("download-annotated").click();
        assert.equal(window.anchorClicks.length, 0); // nothing to export yet
        choose(photo());
        await page.analyseChosenPhoto();
        $("export-json").click();
        const exported = calls.created.at(-1);
        assert.equal(exported.type, "application/json");
        const report = JSON.parse(await exported.text());
        assert.ok(!("annotated_image" in report));
        assert.equal(report.status, "NON_HUMAN_FACE_PRESENT");
        assert.match(window.anchorClicks[0].download, /^photo-analysis-\d+\.json$/);
        assert.ok(calls.revoked.includes(`blob:${calls.created.length}`));

        $("download-annotated").click();
        assert.match(window.anchorClicks[1].download, /^annotated-photo-\d+\.jpg$/);
        assert.match(window.anchorClicks[1].href, /^data:image\/jpeg/);
        page.state.result.annotated_image = null;
        $("download-annotated").click();
        assert.equal(window.anchorClicks.length, 2);
    });

    it("redraws the boxes whenever the photo changes size", async () => {
        const { deps } = fakes();
        const page = init(deps);
        FakeObserver.last.callback(); // no result yet: nothing happens
        choose(photo());
        giveSize($("original-image"));
        await page.analyseChosenPhoto();
        $("overlay").replaceChildren();
        FakeObserver.last.callback();
        assert.equal($("overlay").children.length, 3);
    });

    it("scrolls smoothly unless the person prefers reduced motion", async () => {
        const { deps } = fakes({ scrollOptions: undefined });
        delete deps.scrollOptions;
        const noMatch = { matches: false };
        window.matchMedia = () => noMatch;
        globalThis.matchMedia = window.matchMedia;
        try {
            const page = init({ ...deps });
            choose(photo());
            await page.analyseChosenPhoto();
            assert.equal($("result-panel").scrolledWith.behavior, "smooth");
            noMatch.matches = true;
            $("analyze-another").click();
            assert.equal(window.scrolledTo.behavior, "auto");
        } finally {
            delete globalThis.matchMedia;
        }
    });
});
