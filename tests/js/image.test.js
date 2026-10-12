import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fitToLimit, MAX_SIDES, QUALITIES } from "../../app/static/js/image.js";
import { loadPage } from "./helpers.js";

const file = (bytes, name = "big.photo.PNG") => new File([new Uint8Array(bytes)], name, { type: "image/png" });
const bitmap = { width: 6000, height: 3000 };
const blobOf = (bytes) => new Blob([new Uint8Array(bytes)]);

describe("fitToLimit", () => {
    it("leaves a photo within the limit alone", async () => {
        const small = file(10);
        assert.deepEqual(await fitToLimit(small, 100), { file: small, shrunk: false });
    });

    it("redraws a large photo smaller, upright, as JPEG, until it fits", async () => {
        const tried = [];
        const toBlob = async (_bitmap, width, height, quality) => {
            tried.push([width, height, quality]);
            return blobOf(tried.length < 3 ? 500 : 50);
        };
        const result = await fitToLimit(file(1000), 100, { createBitmap: async () => bitmap, toBlob });
        assert.equal(result.shrunk, true);
        assert.equal(result.file.name, "big.photo.jpg");
        assert.equal(result.file.type, "image/jpeg");
        assert.deepEqual(tried[0], [3072, 1536, 0.9]);
        assert.deepEqual(tried[2], [3072, 1536, 0.7]);
        assert.equal(result.file.size, 50);
    });

    it("never enlarges a photo that is already smaller than the maximum side", async () => {
        const tried = [];
        const toBlob = async (_bitmap, width, height, quality) => {
            tried.push([width, height, quality]);
            return blobOf(tried.length < 2 ? 500 : 10);
        };
        await fitToLimit(file(1000), 100, { createBitmap: async () => ({ width: 1200, height: 800 }), toBlob });
        assert.deepEqual(tried, [
            [1200, 800, 0.9],
            [1200, 800, 0.8],
        ]);
    });

    it("tries every size before giving up", async () => {
        const tried = new Set();
        const toBlob = async (_bitmap, width) => {
            tried.add(width);
            return blobOf(500);
        };
        await assert.rejects(fitToLimit(file(1000), 100, { createBitmap: async () => bitmap, toBlob }), /too large even after shrinking/);
        assert.deepEqual([...tried], MAX_SIDES);
        assert.equal(QUALITIES.length, 3);
    });

    it("explains when the browser cannot read the photo", async () => {
        const createBitmap = async () => {
            throw new Error("nope");
        };
        await assert.rejects(
            fitToLimit(file(1000), 100, { createBitmap }),
            (error) => error.code === "file_too_large" && /could not shrink/.test(error.message),
        );
    });

    it("gives up when no image data is produced", async () => {
        await assert.rejects(fitToLimit(file(1000), 100, { createBitmap: async () => bitmap, toBlob: async () => null }), /too large/);
    });

    it("draws with a canvas by default", async () => {
        loadPage();
        const drawn = [];
        const original = document.createElement.bind(document);
        document.createElement = (tag) => {
            const node = original(tag);
            if (tag === "canvas") {
                node.getContext = () => ({ drawImage: (...args) => drawn.push(args.slice(1)) });
                node.toBlob = (callback) => callback(blobOf(10));
            }
            return node;
        };
        globalThis.createImageBitmap = async () => ({ width: 100, height: 50 });
        try {
            const result = await fitToLimit(file(1000), 100);
            assert.equal(result.shrunk, true);
            assert.deepEqual(drawn[0], [0, 0, 100, 50]);
        } finally {
            document.createElement = original;
            delete globalThis.createImageBitmap;
        }
    });
});
