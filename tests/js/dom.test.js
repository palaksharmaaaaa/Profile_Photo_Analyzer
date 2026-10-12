import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";
import { $, chip, el, hide, setText, show } from "../../app/static/js/dom.js";
import { loadPage } from "./helpers.js";

describe("dom", () => {
    beforeEach(() => loadPage());

    it("builds elements with text, attributes and children, and never as HTML", () => {
        const child = el("span", { text: "inner" });
        const node = el("div", { class: "card", id: "x", text: "<b>hi</b>", attrs: { role: "group" } }, child, null);
        assert.equal(node.className, "card");
        assert.equal(node.id, "x");
        assert.equal(node.getAttribute("role"), "group");
        assert.equal(node.querySelector("b"), null);
        assert.equal(node.firstChild.nodeValue, "<b>hi</b>");
        assert.equal(node.children.length, 1);
        assert.equal(el("p").outerHTML, "<p></p>");
    });

    it("shows and hides elements of the page", () => {
        hide("result-panel");
        assert.ok($("result-panel").classList.contains("hidden"));
        show("result-panel");
        assert.ok(!$("result-panel").classList.contains("hidden"));
    });

    it("writes text or a fallback", () => {
        setText("reason", "because");
        assert.equal($("reason").textContent, "because");
        setText("reason", null);
        assert.equal($("reason").textContent, "-");
        setText("reason", "", "none");
        assert.equal($("reason").textContent, "none");
        setText("reason", 0);
        assert.equal($("reason").textContent, "0");
    });

    it("makes chips with a tone", () => {
        assert.equal(chip("a").className, "chip");
        assert.equal(chip("a", "red").className, "chip red");
    });
});
