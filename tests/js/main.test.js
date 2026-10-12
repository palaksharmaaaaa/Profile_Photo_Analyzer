import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { $ } from "../../app/static/js/dom.js";
import { loadPage } from "./helpers.js";

describe("main", () => {
    it("starts the page when it is loaded as the entry point", async () => {
        loadPage();
        globalThis.ResizeObserver = class {
            observe() {}
        };
        try {
            await import("../../app/static/js/main.js");
            $("photo").dispatchEvent(new window.Event("change"));
            assert.equal($("upload-form").tagName, "FORM");
            assert.ok($("reset-button").classList.contains("hidden"));
        } finally {
            delete globalThis.ResizeObserver;
        }
    });
});
