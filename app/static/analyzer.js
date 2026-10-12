/* Profile Photo Analyzer - browser script.
 *
 * Uploads the chosen photo to POST /analyze-profile-photo and renders the JSON response
 * (see app/schemas/analysis.py for every field). All text from the server is inserted with
 * textContent, never as HTML.
 */
"use strict";

(() => {
    const $ = (id) => document.getElementById(id);

    // ---- Verdict wording for each status the API can return ------------------------------------------
    const VERDICTS = {
        PASSED: ["Passed all checks", "success"],
        NO_INFORMATION: ["No visual information in this image", "danger"],
        NO_HUMAN_FACE: ["No human face was found", "danger"],
        ONLY_NON_HUMAN_FACES: ["Only non-human faces were found", "warning"],
        MULTIPLE_HUMAN_FACES: ["More than one human face", "danger"],
        NON_HUMAN_FACE_PRESENT: ["A non-human face is also in the photo", "warning"],
        REVIEW_REQUIRED: ["Manual review required", "warning"],
        POOR_FACE_QUALITY: ["The face quality is not good enough", "danger"],
        AGE_UNSURE: ["The age could not be determined", "warning"],
        POSSIBLE_MINOR: ["Age verification required (possible minor)", "warning"],
    };

    const STAGE_BOX_CLASS = { human: "human", other: "other", unsure: "unsure" };
    const AVATAR_SIZE = 48;

    let lastResult = null;
    let previewUrl = null;

    // ---- Small helpers -------------------------------------------------------------------------------
    const show = (id) => $(id).classList.remove("hidden");
    const hide = (id) => $(id).classList.add("hidden");

    /** Create an element: el("div", {class: "card", text: "Hello"}, child1, child2). */
    function el(tag, options = {}, ...children) {
        const node = document.createElement(tag);
        if (options.class) node.className = options.class;
        if (options.text !== undefined) node.textContent = options.text;
        if (options.id) node.id = options.id;
        children.forEach((child) => child && node.appendChild(child));
        return node;
    }

    const percent = (value, digits = 0) =>
        value === null || value === undefined ? null : `${(value * 100).toFixed(digits)}%`;

    const setText = (id, value) => {
        $(id).textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    };

    function chip(text, tone = "") {
        return el("span", { class: `chip ${tone}`.trim(), text });
    }

    // ---- Pipeline stepper ----------------------------------------------------------------------------
    function setStep(number, status) {
        const step = document.querySelector(`.step[data-step="${number}"]`);
        const pill = step.querySelector(".pill");
        pill.className = `pill pill-${status}`;
        pill.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        step.classList.toggle("active", status === "running");
    }

    function resetSteps() {
        [1, 2, 3, 4].forEach((n) => setStep(n, "pending"));
    }

    // ---- Rendering a result --------------------------------------------------------------------------
    function renderVerdict(data) {
        const [text, tone] = VERDICTS[data.status] || [data.status, "neutral"];
        $("verdict").textContent = text;
        $("verdict").className = `verdict verdict-${tone}`;
        setText("reason", data.reason);
        setText("summary", data.summary);
    }

    function renderCards(data) {
        const info = data.image_info;
        setText("info-value", info.has_information ? "Has content" : "No information");
        setText("info-hint", `Entropy ${info.entropy}, std-dev ${info.std_dev}`);

        const type = data.image_type;
        setText("type-value", type.label);
        setText("type-hint", type.photo_share === null ? type.reason : `${percent(type.photo_share)} photographic`);

        const humans = data.real_human_face_count;
        setText("human-count", humans);
        setText("human-hint", humans === 1 ? "1 verified real human face" : "Verified by the models");

        const others = data.other_face_count;
        setText("other-count", others);
        setText("other-hint", data.unsure_face_count ? `${data.unsure_face_count} more unverified` : "");
    }

    /** A 48px thumbnail of the face, cropped from the returned (analysed) image. */
    function avatar(face) {
        const canvas = el("canvas", { class: "avatar" });
        canvas.width = canvas.height = AVATAR_SIZE;
        const source = $("original-image");
        const draw = () => {
            const [x1, y1, x2, y2] = face.box_xyxy;
            canvas.getContext("2d").drawImage(source, x1, y1, x2 - x1, y2 - y1, 0, 0, AVATAR_SIZE, AVATAR_SIZE);
        };
        source.complete ? draw() : source.addEventListener("load", draw, { once: true });
        return canvas;
    }

    function ageColumn(face) {
        const column = el("div", {}, el("div", { class: "card-label", text: "Age (3-year bracket)" }));
        column.appendChild(el("div", { class: "card-value", text: face.age_group }));
        if (face.age_status === "Estimated") {
            const [low, high] = face.age_range;
            column.appendChild(el("div", {
                class: "card-hint",
                text: `Estimated ${face.estimated_age} years, 80% range ${low} to ${high}. ` +
                      `Probability of this bracket: ${percent(face.age_group_probability)}.`,
            }));
        } else {
            column.appendChild(el("div", { class: "card-hint", text: face.notes.join(" ") || "The models could not support an age." }));
        }
        if (face.p_under_18 !== null) {
            const label = `Under 18: ${percent(face.p_under_18)}`;
            column.appendChild(chip(label, face.possible_minor ? "red" : "green"));
        }
        return column;
    }

    function genderColumn(face) {
        const column = el("div", {}, el("div", { class: "card-label", text: "Gender (appearance-based)" }));
        column.appendChild(el("div", { class: "card-value", text: face.gender }));
        const hint = face.gender_confidence === null ? "" : `Model score ${percent(face.gender_confidence, 1)}`;
        column.appendChild(el("div", { class: "card-hint", text: hint }));
        return column;
    }

    function qualityColumn(face) {
        const quality = face.quality;
        const column = el("div", {}, el("div", { class: "card-label", text: "Framing and quality" }));
        column.appendChild(el("div", { class: "card-value", text: quality.status.replaceAll("_", " ").toLowerCase() }));
        column.appendChild(el("div", { class: "card-hint", text: quality.reason }));
        return column;
    }

    function humanCard(face) {
        const title = el("div", {},
            el("h3", { text: `Face #${face.face_id}` }),
            el("div", {
                class: "card-hint",
                text: `Detection ${percent(face.detection_confidence, 1)}, verified human ${percent(face.verification_confidence, 1)}`,
            }));
        const card = el("div", { class: "face-card", id: `human-card-${face.face_id}` },
            el("div", { class: "face-head" }, avatar(face), title),
            el("div", { class: "grid-3" }, ageColumn(face), genderColumn(face), qualityColumn(face)));
        linkHover(card, `human-${face.face_id}`);
        return card;
    }

    function otherChip(face) {
        const source = face.verified_by ? `, checked by ${face.verified_by}` : "";
        const score = face.confidence === null ? "" : ` (${percent(face.confidence, 0)})`;
        const node = chip(`${face.label}${score}${source}`, "purple");
        node.id = `other-chip-${face.face_id}`;
        linkHover(node, `other-${face.face_id}`);
        return node;
    }

    function unsureChip(face) {
        const node = chip(`${face.label}: ${face.reason}`, "amber");
        node.id = `unsure-chip-${face.face_id}`;
        linkHover(node, `unsure-${face.face_id}`);
        return node;
    }

    function renderFaceLists(data) {
        const fill = (sectionId, containerId, items, build) => {
            const container = $(containerId);
            container.replaceChildren(...items.map(build));
            $(sectionId).classList.toggle("hidden", items.length === 0);
        };
        fill("humans-section", "humans", data.human_faces, humanCard);
        fill("others-section", "others", data.other_faces, otherChip);
        fill("unsure-section", "unsure", data.unsure_faces, unsureChip);
    }

    // ---- Overlay boxes on the original photo ---------------------------------------------------------
    /** Highlight the box and the list entry together when either is hovered. */
    function linkHover(listNode, boxId) {
        const toggle = (on) => {
            const box = $(`box-${boxId}`);
            if (box) box.classList.toggle("active", on);
            listNode.classList.toggle("active", on);
        };
        listNode.addEventListener("mouseenter", () => toggle(true));
        listNode.addEventListener("mouseleave", () => toggle(false));
    }

    function drawOverlay(data) {
        const overlay = $("overlay");
        const image = $("original-image");
        overlay.replaceChildren();
        if (!image.naturalWidth) return;

        const entries = [
            ...data.human_faces.map((f) => ["human", f, `Face #${f.face_id} (${f.gender})`, `human-card-${f.face_id}`]),
            ...data.other_faces.map((f) => ["other", f, f.label, `other-chip-${f.face_id}`]),
            ...data.unsure_faces.map((f) => ["unsure", f, "Unsure", `unsure-chip-${f.face_id}`]),
        ];
        for (const [kind, face, label, listId] of entries) {
            const [x1, y1, x2, y2] = face.box_percent;
            const box = el("div", { class: `box ${STAGE_BOX_CLASS[kind]}`, id: `box-${kind}-${face.face_id}` }, el("span", { class: "box-tag", text: label }));
            box.style.left = `${image.offsetLeft + (x1 / 100) * image.clientWidth}px`;
            box.style.top = `${image.offsetTop + (y1 / 100) * image.clientHeight}px`;
            box.style.width = `${((x2 - x1) / 100) * image.clientWidth}px`;
            box.style.height = `${((y2 - y1) / 100) * image.clientHeight}px`;
            const listNode = $(listId);
            const toggle = (on) => {
                box.classList.toggle("active", on);
                if (listNode) listNode.classList.toggle("active", on);
            };
            box.addEventListener("mouseenter", () => toggle(true));
            box.addEventListener("mouseleave", () => toggle(false));
            overlay.appendChild(box);
        }
    }

    // ---- Technical details ---------------------------------------------------------------------------
    function renderTechnical(data) {
        const d = data.diagnostics;
        const original = d.original_width ? `${d.original_width} x ${d.original_height} px` : "-";
        const adjustments = [d.exif_orientation_applied && "rotated by its EXIF tag", d.downscaled_for_analysis && "downscaled for analysis"]
            .filter(Boolean).join(", ");
        const rows = [
            ["Processing time", `${d.processing_time_ms} ms`],
            ["Analysed size", `${d.image_width} x ${d.image_height} px`],
            ["Uploaded size", adjustments ? `${original} (${adjustments})` : original],
            ["File format", d.file_format],
            ["Image type", data.image_type.model || data.image_type.reason],
            ["False detections discarded", d.rejected_candidate_count],
            ["Models that ran", d.models_executed.join(", ")],
        ];
        $("technical-rows").replaceChildren(...rows.map(([name, value]) => el("tr", {}, el("th", { text: name }), el("td", { text: value }))));
        setText("timing", `${d.processing_time_ms} ms`);
    }

    function renderResult(data) {
        lastResult = data;
        data.pipeline_steps.forEach((step) => setStep(step.step_number, step.status));
        renderVerdict(data);
        renderCards(data);

        const original = $("original-image");
        original.onload = () => drawOverlay(data);
        original.src = data.original_image;
        $("annotated-image").src = data.annotated_image;
        setText("filename", data.filename);
        setText("legend", data.legend);
        setText("disclaimer", data.disclaimer);

        renderFaceLists(data);
        renderTechnical(data);
        if (original.complete) drawOverlay(data);
    }

    // ---- Upload flow ---------------------------------------------------------------------------------
    function showPreview(file) {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        $("loading-preview").src = previewUrl;
    }

    async function analyze(event) {
        event.preventDefault();
        const form = $("upload-form");
        const file = $("photo").files[0];
        if (!file) return;

        hide("result-panel");
        hide("error-panel");
        show("loading-panel");
        $("analyze-button").disabled = true;
        $("analyze-button").textContent = "Analyzing...";
        resetSteps();
        setStep(1, "running");
        setStep(2, "running");

        try {
            const response = await fetch("/analyze-profile-photo", { method: "POST", body: new FormData(form) });
            const data = await response.json();
            if (!response.ok) {
                setStep(1, "failed");
                throw new Error(data.reason || data.detail || `The server answered with status ${response.status}.`);
            }
            show("result-panel");
            renderResult(data);
            show("reset-button");
        } catch (error) {
            $("error-text").textContent = error.message;
            show("error-panel");
        } finally {
            hide("loading-panel");
            $("analyze-button").disabled = false;
            $("analyze-button").textContent = "Analyze photo";
        }
    }

    function reset() {
        $("upload-form").reset();
        lastResult = null;
        ["result-panel", "error-panel", "loading-panel", "reset-button"].forEach(hide);
        resetSteps();
    }

    function download(href, name) {
        const link = el("a");
        link.href = href;
        link.download = name;
        link.click();
    }

    // ---- Wiring --------------------------------------------------------------------------------------
    const zone = $("drop-zone");
    ["dragenter", "dragover"].forEach((name) =>
        zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((name) =>
        zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragover"); }));
    zone.addEventListener("drop", (event) => {
        if (event.dataTransfer.files.length) {
            $("photo").files = event.dataTransfer.files;
            showPreview(event.dataTransfer.files[0]);
        }
    });
    $("photo").addEventListener("change", (event) => event.target.files[0] && showPreview(event.target.files[0]));
    $("upload-form").addEventListener("submit", analyze);
    $("reset-button").addEventListener("click", reset);
    $("analyze-another").addEventListener("click", reset);
    $("export-json").addEventListener("click", () => {
        if (!lastResult) return;
        const report = { ...lastResult, original_image: undefined, annotated_image: undefined };
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
        download(url, `photo-analysis-${Date.now()}.json`);
        URL.revokeObjectURL(url);
    });
    $("download-annotated").addEventListener("click", () => {
        if (lastResult) download(lastResult.annotated_image, `annotated-photo-${Date.now()}.jpg`);
    });
    // Keep the boxes aligned with the photo whenever its displayed size changes (window resize, panel shown, ...).
    new ResizeObserver(() => lastResult && drawOverlay(lastResult)).observe($("original-image"));
})();
