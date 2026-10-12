// The behaviour of the page: choose a photo, analyse it, show the result, export it.

import { analyse, ApiProblem } from "./api.js";
import { $, el, hide, show } from "./dom.js";
import { downloadName, megabytes } from "./format.js";
import { fitToLimit } from "./image.js";
import { applyStage, drawOverlay, renderResult, resetSteps, setStep, showBusy, showError } from "./render.js";

export const LIMITS_URL = "/v1/limits";

/** What a failed analysis says to the person, including when to try again. */
export function describeProblem(error) {
    if (!(error instanceof ApiProblem)) {
        return "The server could not be reached. Check the connection and try again.";
    }
    const wait = error.retryAfter ? ` Try again in ${error.retryAfter} seconds.` : "";
    const reference = error.requestId ? ` (reference ${error.requestId})` : "";
    return `${error.message}${wait}${reference}`;
}

/** Everything the page takes from the browser; tests pass replacements. */
function browserDependencies() {
    return {
        analyseImpl: analyse,
        fitImpl: fitToLimit,
        fetchImpl: globalThis.fetch?.bind(globalThis),
        createObjectUrl: (blob) => URL.createObjectURL(blob),
        revokeObjectUrl: (url) => URL.revokeObjectURL(url),
        ResizeObserverImpl: globalThis.ResizeObserver,
        scrollOptions: () => ({
            behavior: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        }),
    };
}

// ---- The drop zone ----------------------------------------------------------------------------------------

function wireDropZone(onFile) {
    const zone = $("drop-zone");
    for (const name of ["dragenter", "dragover"]) {
        zone.addEventListener(name, (event) => {
            event.preventDefault();
            zone.classList.add("dragover");
        });
    }
    for (const name of ["dragleave", "drop"]) {
        zone.addEventListener(name, (event) => {
            event.preventDefault();
            zone.classList.remove("dragover");
        });
    }
    zone.addEventListener("drop", (event) => {
        if (event.dataTransfer.files.length) {
            $("photo").files = event.dataTransfer.files;
            onFile(event.dataTransfer.files[0]);
        }
    });
    $("photo").addEventListener("change", (event) => event.target.files[0] && onFile(event.target.files[0]));
}

function resetDropZone(defaultHint) {
    hide("drop-preview");
    show("upload-visual");
    $("upload-title-text").replaceChildren("Drop a photo here, or ", el("span", { class: "link", text: "browse" }));
    $("upload-hint").textContent = defaultHint;
}

function download(href, name) {
    const link = el("a");
    link.href = href;
    link.download = name;
    link.click();
}

// ---- The page ---------------------------------------------------------------------------------------------

/**
 * Attach the page's behaviour. Returns the pieces tests need: {state, analyseChosenPhoto, reset, loadLimits}.
 * `overrides` replaces anything from `browserDependencies()`.
 */
export function init(overrides = {}) {
    const deps = { ...browserDependencies(), ...overrides };
    const state = { result: null, photoUrl: null, controller: null, maxBytes: null };
    const defaultHint = $("upload-hint").textContent;

    async function loadLimits() {
        try {
            const response = await deps.fetchImpl(LIMITS_URL);
            if (response.ok) {
                const limits = await response.json();
                state.maxBytes = limits.max_upload_bytes;
                $("upload-hint").textContent = `${defaultHint} - up to ${megabytes(limits.max_upload_bytes)} MB`;
            }
        } catch {
            // the page still works; a photo over the limit is then refused by the server
        }
    }

    function rememberPhoto(file) {
        if (state.photoUrl) {
            deps.revokeObjectUrl(state.photoUrl);
        }
        state.photoUrl = deps.createObjectUrl(file);
        $("loading-preview").src = state.photoUrl;
    }

    function showChosenFile(file) {
        rememberPhoto(file);
        $("drop-preview").src = state.photoUrl;
        show("drop-preview");
        hide("upload-visual");
        $("upload-title-text").textContent = file.name;
        $("upload-hint").textContent = `${megabytes(file.size)} MB - click to choose a different photo`;
    }

    async function prepare(chosen) {
        if (!state.maxBytes) {
            return chosen;
        }
        const { file, shrunk } = await deps.fitImpl(chosen, state.maxBytes);
        if (shrunk) {
            rememberPhoto(file);
        }
        return file;
    }

    function showResult(result) {
        state.result = result;
        show("result-panel");
        renderResult(result, state.photoUrl);
        show("reset-button");
        $("result-panel").scrollIntoView({ block: "start", ...deps.scrollOptions() });
    }

    async function analyseChosenPhoto(event) {
        event?.preventDefault();
        const chosen = $("photo").files[0];
        if (!chosen) {
            return;
        }
        state.controller?.abort();
        const controller = new AbortController();
        state.controller = controller;

        showBusy(true);
        resetSteps();
        setStep(1, "running");
        try {
            const file = await prepare(chosen);
            controller.signal.throwIfAborted(); // a newer analysis has taken over while the photo was being prepared
            showResult(await deps.analyseImpl(file, applyStage, { signal: controller.signal, fetchImpl: deps.fetchImpl }));
        } catch (error) {
            if (error.name !== "AbortError") {
                setStep(1, "failed");
                showError(describeProblem(error));
            }
        } finally {
            if (state.controller === controller) {
                showBusy(false);
            }
        }
    }

    function reset() {
        state.controller?.abort();
        $("upload-form").reset();
        state.result = null;
        ["result-panel", "error-panel", "loading-panel", "reset-button"].forEach(hide);
        resetDropZone(defaultHint);
        resetSteps();
        window.scrollTo({ top: 0, ...deps.scrollOptions() });
    }

    function exportJson() {
        if (!state.result) {
            return;
        }
        const { annotated_image: _omitted, ...report } = state.result;
        const url = deps.createObjectUrl(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
        download(url, downloadName("photo-analysis", "json"));
        deps.revokeObjectUrl(url);
    }

    function downloadAnnotated() {
        if (state.result?.annotated_image) {
            download(state.result.annotated_image, downloadName("annotated-photo", "jpg"));
        }
    }

    wireDropZone(showChosenFile);
    $("upload-form").addEventListener("submit", analyseChosenPhoto);
    $("reset-button").addEventListener("click", reset);
    $("analyze-another").addEventListener("click", reset);
    $("export-json").addEventListener("click", exportJson);
    $("download-annotated").addEventListener("click", downloadAnnotated);
    // Keep the boxes aligned with the photo whenever its displayed size changes (window resize, panel shown, ...).
    new deps.ResizeObserverImpl(() => state.result && drawOverlay(state.result)).observe($("original-image"));

    loadLimits();
    return { state, analyseChosenPhoto, reset, loadLimits };
}
