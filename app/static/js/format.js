// Pure helpers: wording and number formatting. Nothing here touches the page.

/** Heading and tone for each outcome the API can return. */
export const VERDICTS = {
    PASSED: ["Passed all checks", "success"],
    NO_INFORMATION: ["No visual information in this image", "danger"],
    NO_HUMAN_FACE: ["No human face was found", "danger"],
    ONLY_NON_HUMAN_FACES: ["Only non-human faces were found", "warning"],
    MULTIPLE_HUMAN_FACES: ["More than one human face", "danger"],
    NON_HUMAN_FACE_PRESENT: ["A non-human face is also in the photo", "warning"],
    REVIEW_REQUIRED: ["Manual review required", "warning"],
    POOR_FACE_QUALITY: ["The face quality is not good enough", "danger"],
    AGE_UNSURE: ["The age could not be determined", "warning"],
    POSSIBLE_MINOR: ["Age check needed (possible minor)", "warning"],
};

export const verdictFor = (status) => VERDICTS[status] ?? [status, "neutral"];

/** 0.873 -> "87%" (or "87.3%" with one digit); null and undefined give null. */
export function percent(value, digits = 0) {
    if (value === null || value === undefined) {
        return null;
    }
    return `${(value * 100).toFixed(digits)}%`;
}

/** "-" for anything that has no value to show. */
export function orDash(value) {
    return value === null || value === undefined || value === "" ? "-" : String(value);
}

export const megabytes = (bytes) => (bytes / 1024 / 1024).toFixed(2);

export const lowerWords = (code) => code.replaceAll("_", " ").toLowerCase();

export const capitalise = (word) => word.charAt(0).toUpperCase() + word.slice(1);

/** The sentence under the age bracket of one face. */
export function ageDetail(face) {
    if (face.age_status !== "Estimated") {
        return face.notes.join(" ") || "The models could not support an age.";
    }
    const [low, high] = face.age_range;
    return (
        `Estimated ${face.estimated_age} years, 80% range ${low} to ${high}. ` +
        `Probability of this bracket: ${percent(face.age_group_probability)}.`
    );
}

/** The sentence under the gender of one face. */
export function genderDetail(face) {
    return face.gender_confidence === null ? "" : `Model score ${percent(face.gender_confidence, 1)}`;
}

/** Position of a box given in percent inside an element of the given pixel size. */
export function boxStyle(boxPercent, width, height, offsetLeft = 0, offsetTop = 0) {
    const [x1, y1, x2, y2] = boxPercent;
    return {
        left: offsetLeft + (x1 / 100) * width,
        top: offsetTop + (y1 / 100) * height,
        width: ((x2 - x1) / 100) * width,
        height: ((y2 - y1) / 100) * height,
    };
}

/** "pho.png" -> "photo-analysis-1700000000000.json" style names for downloads. */
export const downloadName = (prefix, extension, now = Date.now()) => `${prefix}-${now}.${extension}`;
