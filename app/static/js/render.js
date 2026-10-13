// Drawing the page from API data. Every function reads the data it is given and writes only to the page.

import { $, chip, el, hide, setText, show } from "./dom.js";
import { ageDetail, boxStyle, capitalise, genderDetail, lowerWords, orDash, percent, verdictFor } from "./format.js";

const AVATAR_SIZE = 52;
const STEP_NUMBERS = [1, 2, 3, 4];

// ---- The four pipeline steps ------------------------------------------------------------------------------

export function setStep(number, status) {
    const step = document.querySelector(`.step[data-step="${number}"]`);
    const pill = step.querySelector(".pill");
    pill.className = `pill pill-${status}`;
    pill.textContent = capitalise(status);
    step.classList.toggle("active", status === "running");
    step.setAttribute("aria-current", status === "running" ? "step" : "false");
}

export function resetSteps() {
    STEP_NUMBERS.forEach((number) => setStep(number, "pending"));
}

/** A stage event from the job (step_number, status, ...) changes one step. */
export function applyStage(stage) {
    setStep(stage.step_number, stage.status);
}

// ---- Verdict and summary cards ----------------------------------------------------------------------------

export function renderVerdict(data) {
    const [text, tone] = verdictFor(data.status);
    $("verdict").textContent = text;
    $("verdict").className = `verdict verdict-${tone}`;
    setText("reason", data.reason);
    setText("summary", data.summary);
}

export function renderCards(data) {
    const info = data.image_info;
    setText("info-value", info.has_information ? "Has content" : "No information");
    setText("info-hint", `Entropy ${info.entropy}, std-dev ${info.std_dev}`);

    const type = data.image_type;
    setText("type-value", type.label);
    setText("type-hint", type.photo_share === null ? type.reason : `${percent(type.photo_share)} photographic`);

    setText("human-count", data.real_human_face_count);
    setText("human-hint", data.real_human_face_count === 1 ? "1 verified real human face" : "Verified by the models");

    setText("other-count", data.other_face_count);
    setText("other-hint", data.unsure_face_count ? `${data.unsure_face_count} more unverified` : "");
}

// ---- Face cards -------------------------------------------------------------------------------------------

/** A square thumbnail of the face, cut from the photo the person chose (the box is in percent of that photo). */
export function avatar(face, source) {
    const canvas = el("canvas", { class: "avatar", attrs: { "aria-hidden": "true" } });
    canvas.width = AVATAR_SIZE;
    canvas.height = AVATAR_SIZE;
    const draw = () => {
        const [x1, y1, x2, y2] = face.box_percent;
        const sx = (x1 / 100) * source.naturalWidth;
        const sy = (y1 / 100) * source.naturalHeight;
        const sw = ((x2 - x1) / 100) * source.naturalWidth;
        const sh = ((y2 - y1) / 100) * source.naturalHeight;
        canvas.getContext("2d")?.drawImage(source, sx, sy, sw, sh, 0, 0, AVATAR_SIZE, AVATAR_SIZE);
    };
    if (source.complete && source.naturalWidth) {
        draw();
    } else {
        source.addEventListener("load", draw, { once: true });
    }
    return canvas;
}

function column(label, value, hint, extra) {
    const node = el("div", {}, el("div", { class: "card-label", text: label }));
    node.appendChild(el("div", { class: "card-value", text: value }));
    if (hint) {
        node.appendChild(el("div", { class: "card-hint", text: hint }));
    }
    if (extra) {
        node.appendChild(extra);
    }
    return node;
}

function underEighteenChip(face) {
    if (face.p_under_18 === null) {
        return null;
    }
    const text = face.possible_minor ? `Under 18: ${percent(face.p_under_18)} - age check needed` : `Under 18: ${percent(face.p_under_18)}`;
    return chip(text, face.possible_minor ? "red" : "green");
}

/** Highlight a list entry and its box on the photo together. */
export function linkHover(listNode, boxId) {
    const toggle = (on) => {
        $(`box-${boxId}`)?.classList.toggle("active", on);
        listNode.classList.toggle("active", on);
    };
    listNode.addEventListener("mouseenter", () => toggle(true));
    listNode.addEventListener("mouseleave", () => toggle(false));
}

export function humanCard(face, source) {
    const title = el(
        "div",
        {},
        el("h3", { text: `Face #${face.face_id}` }),
        el("div", {
            class: "card-hint",
            text: `Detection ${percent(face.detection_confidence, 1)}, verified human ${percent(face.verification_confidence, 1)}`,
        }),
    );
    const details = el(
        "div",
        { class: "grid-3" },
        column("Age (3-year bracket)", face.age_group, ageDetail(face), underEighteenChip(face)),
        column("Gender (appearance-based)", face.gender, genderDetail(face)),
        column("Framing and quality", lowerWords(face.quality.status), face.quality.reason),
    );
    const card = el(
        "div",
        { class: "face-card", id: `human-card-${face.face_id}` },
        el("div", { class: "face-head" }, avatar(face, source), title),
        details,
    );
    linkHover(card, `human-${face.face_id}`);
    return card;
}

export function otherChip(face) {
    const source = face.verified_by ? `, checked by ${face.verified_by}` : "";
    const score = face.confidence === null ? "" : ` (${percent(face.confidence)})`;
    const node = chip(`${face.label}${score}${source}`, "purple");
    node.id = `other-chip-${face.face_id}`;
    linkHover(node, `other-${face.face_id}`);
    return node;
}

export function unsureChip(face) {
    const node = chip(`${face.label}: ${face.reason}`, "amber");
    node.id = `unsure-chip-${face.face_id}`;
    linkHover(node, `unsure-${face.face_id}`);
    return node;
}

function fill(sectionId, containerId, items, build) {
    $(containerId).replaceChildren(...items.map(build));
    $(sectionId).classList.toggle("hidden", items.length === 0);
}

export function renderFaceLists(data, source) {
    fill("humans-section", "humans", data.human_faces, (face) => humanCard(face, source));
    fill("others-section", "others", data.other_faces, otherChip);
    fill("unsure-section", "unsure", data.unsure_faces, unsureChip);
}

// ---- Boxes over the photo ---------------------------------------------------------------------------------

function boxEntries(data) {
    return [
        ...data.human_faces.map((face) => ({
            kind: "human",
            face,
            label: `Face #${face.face_id} (${face.gender})`,
            listId: `human-card-${face.face_id}`,
        })),
        ...data.other_faces.map((face) => ({ kind: "other", face, label: face.label, listId: `other-chip-${face.face_id}` })),
        ...data.unsure_faces.map((face) => ({ kind: "unsure", face, label: "Unsure", listId: `unsure-chip-${face.face_id}` })),
    ];
}

function makeBox({ kind, face, label, listId }, image) {
    const box = el(
        "div",
        { class: `box ${kind}`, id: `box-${kind}-${face.face_id}`, attrs: { tabindex: "0", role: "img", "aria-label": label } },
        el("span", { class: "box-tag", text: label }),
    );
    const position = boxStyle(face.box_percent, image.clientWidth, image.clientHeight, image.offsetLeft, image.offsetTop);
    for (const [property, value] of Object.entries(position)) {
        box.style[property] = `${value}px`;
    }
    const toggle = (on) => {
        box.classList.toggle("active", on);
        $(listId)?.classList.toggle("active", on);
    };
    for (const [start, end] of [["mouseenter", "mouseleave"], ["focus", "blur"]]) {
        box.addEventListener(start, () => toggle(true));
        box.addEventListener(end, () => toggle(false));
    }
    box.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            $(listId)?.scrollIntoView({ block: "center" });
        }
    });
    return box;
}

/** (Re)draw every box on the photo; call again whenever the photo's displayed size changes. */
export function drawOverlay(data) {
    const overlay = $("overlay");
    const image = $("original-image");
    overlay.replaceChildren();
    if (!image.naturalWidth) {
        return;
    }
    overlay.append(...boxEntries(data).map((entry) => makeBox(entry, image)));
}

// ---- Technical details ------------------------------------------------------------------------------------

export function renderTechnical(data) {
    const d = data.diagnostics;
    const original = d.original_width ? `${d.original_width} x ${d.original_height} px` : "-";
    const adjustments = [d.exif_orientation_applied && "rotated by its EXIF tag", d.downscaled_for_analysis && "downscaled for analysis"]
        .filter(Boolean)
        .join(", ");
    const rows = [
        ["Processing time", `${d.processing_time_ms} ms`],
        ["Analysed size", `${d.image_width} x ${d.image_height} px`],
        ["Uploaded size", adjustments ? `${original} (${adjustments})` : original],
        ["File format", d.file_format],
        ["Image type", data.image_type.model || data.image_type.reason],
        ["False detections discarded", d.rejected_candidate_count],
        ["Models that ran", d.models_executed.join(", ")],
    ];
    $("technical-rows").replaceChildren(
        ...rows.map(([name, value]) => el("tr", {}, el("th", { text: name }), el("td", { text: orDash(value) }))),
    );
    setText("timing", `${d.processing_time_ms} ms`);
}

// ---- Whole result and states ------------------------------------------------------------------------------

/** Show a finished analysis. `photoUrl` is the photo the person chose; the server does not send it back. */
export function renderResult(data, photoUrl) {
    data.pipeline_steps.forEach((step) => setStep(step.step_number, step.status));
    renderVerdict(data);
    renderCards(data);

    const original = $("original-image");
    original.onload = () => drawOverlay(data);
    original.src = photoUrl;
    const annotated = $("annotated-image");
    annotated.src = data.annotated_image ?? photoUrl;
    annotated.alt = data.annotated_image ? "Detected faces outlined" : "The upload, without annotation";
    setText("filename", data.filename);
    setText("legend", data.legend);
    setText("disclaimer", data.disclaimer);

    renderFaceLists(data, original);
    renderTechnical(data);
    if (original.complete) {
        drawOverlay(data);
    }
}

export function showError(message) {
    $("error-text").textContent = message;
    show("error-panel");
}

export function showBusy(busy) {
    const button = $("analyze-button");
    button.disabled = busy;
    button.textContent = busy ? "Analyzing..." : "Analyze photo";
    $("upload-form").setAttribute("aria-busy", String(busy));
    if (busy) {
        hide("result-panel");
        hide("error-panel");
        show("loading-panel");
    } else {
        hide("loading-panel");
    }
}
