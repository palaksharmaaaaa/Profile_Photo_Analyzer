// Make a photo that is over the upload limit small enough to send, in the browser, before it is uploaded.

import { ApiProblem } from "./api.js";

export const MAX_SIDES = [3072, 2048, 1536];
export const QUALITIES = [0.9, 0.8, 0.7];

const defaultBitmap = (file) => createImageBitmap(file, { imageOrientation: "from-image" });

function defaultBlob(bitmap, width, height, quality) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}

const jpegName = (name) => `${name.replace(/\.[^.]+$/, "")}.jpg`;

/**
 * Returns {file, shrunk}. A photo within `maxBytes` is returned untouched; a larger one is redrawn smaller (upright,
 * as JPEG) in steps until it fits. Throws an ApiProblem when it cannot be read or will not fit.
 */
export async function fitToLimit(file, maxBytes, { createBitmap = defaultBitmap, toBlob = defaultBlob } = {}) {
    if (file.size <= maxBytes) {
        return { file, shrunk: false };
    }
    let bitmap;
    try {
        bitmap = await createBitmap(file);
    } catch {
        throw new ApiProblem("The photo is over the size limit and this browser could not shrink it.", {
            code: "file_too_large",
        });
    }
    for (const maxSide of MAX_SIDES) {
        const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
        const width = Math.max(1, Math.round(bitmap.width * scale));
        const height = Math.max(1, Math.round(bitmap.height * scale));
        for (const quality of QUALITIES) {
            const blob = await toBlob(bitmap, width, height, quality);
            if (blob && blob.size <= maxBytes) {
                return { file: new File([blob], jpegName(file.name), { type: "image/jpeg" }), shrunk: true };
            }
        }
    }
    throw new ApiProblem("The photo is too large even after shrinking it.", { code: "file_too_large" });
}
