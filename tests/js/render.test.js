import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";
import { $ } from "../../app/static/js/dom.js";
import {
    applyStage,
    avatar,
    drawOverlay,
    humanCard,
    linkHover,
    otherChip,
    renderCards,
    renderFaceLists,
    renderResult,
    renderTechnical,
    renderVerdict,
    resetSteps,
    setStep,
    showBusy,
    showError,
    unsureChip,
} from "../../app/static/js/render.js";
import { fixture, giveSize, loadPage } from "./helpers.js";

const result = () => structuredClone(fixture("result.json"));
const stepPill = (n) => document.querySelector(`.step[data-step="${n}"] .pill`).textContent;

describe("steps", () => {
    beforeEach(() => loadPage());

    it("show the state of each step and mark the running one", () => {
        setStep(2, "running");
        const step = document.querySelector('.step[data-step="2"]');
        assert.equal(stepPill(2), "Running");
        assert.ok(step.classList.contains("active"));
        assert.equal(step.getAttribute("aria-current"), "step");
        setStep(2, "completed");
        assert.equal(stepPill(2), "Completed");
        assert.ok(!step.classList.contains("active"));
        assert.equal(step.getAttribute("aria-current"), "false");
    });

    it("start again from pending and follow stage events", () => {
        setStep(1, "completed");
        resetSteps();
        assert.deepEqual([1, 2, 3, 4].map(stepPill), ["Pending", "Pending", "Pending", "Pending"]);
        applyStage({ step_number: 3, status: "failed" });
        assert.equal(stepPill(3), "Failed");
    });
});

describe("verdict and cards", () => {
    beforeEach(() => loadPage());

    it("word the outcome and its reason", () => {
        renderVerdict(result());
        assert.equal($("verdict").textContent, "A non-human face is also in the photo");
        assert.equal($("verdict").className, "verdict verdict-warning");
        assert.match($("reason").textContent, /non-human face/);
        assert.match($("summary").textContent, /1 real human face/);
    });

    it("fill the four summary cards", () => {
        renderCards(result());
        assert.equal($("info-value").textContent, "Has content");
        assert.match($("info-hint").textContent, /^Entropy /);
        assert.equal($("type-value").textContent, "Real Photo");
        assert.equal($("type-hint").textContent, "98% photographic");
        assert.equal($("human-count").textContent, "1");
        assert.equal($("human-hint").textContent, "1 verified real human face");
        assert.equal($("other-count").textContent, "1");
        assert.equal($("other-hint").textContent, "1 more unverified");
    });

    it("cope with an image that has no information or no type", () => {
        const data = result();
        data.image_info.has_information = false;
        data.image_type.photo_share = null;
        data.image_type.reason = "Not checked.";
        data.real_human_face_count = 2;
        data.unsure_face_count = 0;
        renderCards(data);
        assert.equal($("info-value").textContent, "No information");
        assert.equal($("type-hint").textContent, "Not checked.");
        assert.equal($("human-hint").textContent, "Verified by the models");
        assert.equal($("other-hint").textContent, "-");
    });

    it("unknown statuses are shown as they are", () => {
        renderVerdict({ ...result(), status: "BRAND_NEW" });
        assert.equal($("verdict").textContent, "BRAND_NEW");
        assert.equal($("verdict").className, "verdict verdict-neutral");
    });
});

describe("face cards", () => {
    beforeEach(() => loadPage());

    it("show age, gender and quality of a human face, and never build HTML from server text", () => {
        const data = result();
        data.human_faces[0].notes = ["<img src=x onerror=alert(1)>"];
        const source = $("original-image");
        giveSize(source);
        const card = humanCard(data.human_faces[0], source);
        assert.equal(card.id, "human-card-1");
        assert.match(card.textContent, /Face #1/);
        assert.match(card.textContent, /33 to 36/);
        assert.match(card.textContent, /Female/);
        assert.match(card.textContent, /likely complete/);
        assert.match(card.textContent, /Under 18: 0%/);
        assert.equal(card.querySelector("img"), null);
        assert.equal(card.querySelector("canvas").getAttribute("aria-hidden"), "true");
    });

    it("flag a possible minor in words, and cope with a missing under-18 score", () => {
        const data = result();
        const source = $("original-image");
        giveSize(source);
        const minor = { ...data.human_faces[0], p_under_18: 0.7, possible_minor: true };
        assert.match(humanCard(minor, source).textContent, /Under 18: 70% - age check needed/);
        assert.equal(humanCard(minor, source).querySelector(".chip.red") !== null, true);
        const none = { ...data.human_faces[0], p_under_18: null };
        assert.doesNotMatch(humanCard(none, source).textContent, /Under 18/);
    });

    it("cut the avatar from the chosen photo, now or once it has loaded", () => {
        const drawn = [];
        const original = window.HTMLCanvasElement.prototype.getContext;
        window.HTMLCanvasElement.prototype.getContext = () => ({ drawImage: (...args) => drawn.push(args.slice(1)) });
        try {
            const source = $("original-image");
            giveSize(source, { natural: [1000, 500] });
            avatar({ box_percent: [10, 20, 30, 60] }, source);
            assert.deepEqual(drawn[0], [100, 100, 200, 200, 0, 0, 52, 52]);

            const later = document.createElement("img");
            Object.defineProperty(later, "complete", { value: false, configurable: true });
            avatar({ box_percent: [0, 0, 50, 50] }, later);
            assert.equal(drawn.length, 1);
            giveSize(later, { natural: [200, 200] });
            later.dispatchEvent(new window.Event("load"));
            assert.deepEqual(drawn[1], [0, 0, 100, 100, 0, 0, 52, 52]);
        } finally {
            window.HTMLCanvasElement.prototype.getContext = original;
        }
    });

    it("do not fail when the canvas cannot draw", () => {
        const source = $("original-image");
        giveSize(source);
        assert.doesNotThrow(() => avatar({ box_percent: [0, 0, 10, 10] }, source));
    });

    it("list the other and unverified faces as chips", () => {
        const data = result();
        const other = otherChip(data.other_faces[0]);
        assert.match(other.textContent, /Cat \(\d+%\), checked by /);
        assert.equal(other.id, "other-chip-1");
        const noScore = otherChip({ ...data.other_faces[0], confidence: null, verified_by: null });
        assert.equal(noScore.textContent, "Cat");
        const unsure = unsureChip(data.unsure_faces[0]);
        assert.equal(unsure.id, "unsure-chip-1");
        assert.match(unsure.textContent, /^Unsure/);
    });

    it("fill the three lists and hide the empty ones", () => {
        const data = result();
        giveSize($("original-image"));
        renderFaceLists(data, $("original-image"));
        assert.equal($("humans").children.length, 1);
        assert.equal($("others").children.length, 1);
        assert.equal($("unsure").children.length, 1);
        assert.ok(!$("humans-section").classList.contains("hidden"));
        renderFaceLists({ ...data, other_faces: [], unsure_faces: [] }, $("original-image"));
        assert.ok($("others-section").classList.contains("hidden"));
        assert.ok($("unsure-section").classList.contains("hidden"));
    });
});

describe("boxes over the photo", () => {
    beforeEach(() => loadPage());

    function drawn() {
        const data = result();
        giveSize($("original-image"), { client: [400, 300], offset: [10, 5] });
        renderFaceLists(data, $("original-image"));
        drawOverlay(data);
        return data;
    }

    it("draw one box per face at the right place, labelled for assistive technology", () => {
        drawn();
        const boxes = [...$("overlay").children];
        assert.deepEqual(boxes.map((box) => box.className), ["box human", "box other", "box unsure"]);
        const human = $("box-human-1");
        assert.equal(human.style.left, "143.32px"); // 10 + 33.33 % of 400
        assert.equal(human.style.top, "104.99px"); // 5 + 33.33 % of 300
        assert.equal(human.getAttribute("aria-label"), "Face #1 (Female)");
        assert.equal(human.getAttribute("tabindex"), "0");
        assert.equal(human.getAttribute("role"), "img");
        assert.equal($("box-unsure-1").textContent, "Unsure");
    });

    it("highlight the box and its list entry together, by mouse or keyboard", () => {
        drawn();
        const box = $("box-human-1");
        const card = $("human-card-1");
        box.dispatchEvent(new window.Event("mouseenter"));
        assert.ok(box.classList.contains("active") && card.classList.contains("active"));
        box.dispatchEvent(new window.Event("mouseleave"));
        assert.ok(!box.classList.contains("active") && !card.classList.contains("active"));
        box.dispatchEvent(new window.Event("focus"));
        assert.ok(box.classList.contains("active") && card.classList.contains("active"));
        box.dispatchEvent(new window.Event("blur"));
        assert.ok(!card.classList.contains("active"));

        box.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Tab" }));
        assert.equal(card.scrolledWith, undefined);
        box.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
        assert.deepEqual(card.scrolledWith, { block: "center" });
    });

    it("highlight the box when its list entry is hovered", () => {
        drawn();
        const card = $("human-card-1");
        card.dispatchEvent(new window.Event("mouseenter"));
        assert.ok($("box-human-1").classList.contains("active"));
        card.dispatchEvent(new window.Event("mouseleave"));
        assert.ok(!$("box-human-1").classList.contains("active"));
        const lonely = document.createElement("div");
        linkHover(lonely, "human-99"); // no such box: nothing breaks
        lonely.dispatchEvent(new window.Event("mouseenter"));
        assert.ok(lonely.classList.contains("active"));
    });

    it("tolerate a missing list entry and draw nothing until the photo has loaded", () => {
        const data = result();
        giveSize($("original-image"));
        drawOverlay(data); // the list has not been rendered yet
        $("box-human-1").dispatchEvent(new window.Event("mouseenter"));
        $("box-human-1").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
        Object.defineProperty($("original-image"), "naturalWidth", { value: 0, configurable: true });
        drawOverlay(data);
        assert.equal($("overlay").children.length, 0);
    });
});

describe("technical details and the whole result", () => {
    beforeEach(() => loadPage());

    it("list what ran, with the adjustments made to the upload", () => {
        const data = result();
        data.diagnostics.original_width = 6000;
        data.diagnostics.original_height = 4000;
        data.diagnostics.exif_orientation_applied = true;
        data.diagnostics.downscaled_for_analysis = true;
        renderTechnical(data);
        const text = $("technical-rows").textContent;
        assert.match(text, /6000 x 4000 px \(rotated by its EXIF tag, downscaled for analysis\)/);
        assert.match(text, /Models that ran/);
        assert.equal($("timing").textContent, `${data.diagnostics.processing_time_ms} ms`);
    });

    it("show a dash when the upload size is unknown", () => {
        const data = result();
        data.diagnostics.original_width = null;
        data.image_type.model = null;
        renderTechnical(data);
        assert.match($("technical-rows").textContent, /Uploaded size-/);
    });

    it("render everything, using the chosen photo for the overlay and the annotated image from the server", () => {
        const data = result();
        giveSize($("original-image"));
        renderResult(data, "blob:photo");
        assert.equal($("original-image").getAttribute("src"), "blob:photo");
        assert.equal($("annotated-image").getAttribute("src"), data.annotated_image);
        assert.equal($("annotated-image").alt, "Detected faces outlined");
        assert.equal($("filename").textContent, "p.jpg");
        assert.equal($("overlay").children.length, 3);
        assert.deepEqual([1, 2, 3, 4].map(stepPill), ["Completed", "Completed", "Completed", "Completed"]);
    });

    it("draw the overlay when the photo finishes loading, and show the photo when annotation was left out", () => {
        const data = result();
        data.annotated_image = null;
        renderResult(data, "blob:photo");
        assert.equal($("annotated-image").getAttribute("src"), "blob:photo");
        assert.equal($("annotated-image").alt, "The upload, without annotation");
        assert.equal($("overlay").children.length, 0);
        Object.defineProperty($("original-image"), "complete", { value: false, configurable: true });
        giveSize($("original-image"));
        Object.defineProperty($("original-image"), "complete", { value: false, configurable: true });
        $("original-image").onload();
        assert.equal($("overlay").children.length, 3);
    });
});

describe("states", () => {
    beforeEach(() => loadPage());

    it("show an error", () => {
        showError("Something broke.");
        assert.equal($("error-text").textContent, "Something broke.");
        assert.ok(!$("error-panel").classList.contains("hidden"));
    });

    it("switch the page between busy and idle", () => {
        $("result-panel").classList.remove("hidden");
        showBusy(true);
        assert.ok($("analyze-button").disabled);
        assert.equal($("analyze-button").textContent, "Analyzing...");
        assert.equal($("upload-form").getAttribute("aria-busy"), "true");
        assert.ok($("result-panel").classList.contains("hidden"));
        assert.ok(!$("loading-panel").classList.contains("hidden"));
        showBusy(false);
        assert.ok(!$("analyze-button").disabled);
        assert.equal($("analyze-button").textContent, "Analyze photo");
        assert.ok($("loading-panel").classList.contains("hidden"));
        assert.equal($("upload-form").getAttribute("aria-busy"), "false");
    });
});
