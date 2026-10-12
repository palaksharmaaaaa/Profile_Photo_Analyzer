// Shared setup for the browser-code tests: the real page in jsdom, and stand-ins for what jsdom lacks.

import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

export const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

/** Load app/templates/index.html (without its script tag) and make its document the global one. */
export function loadPage() {
    const html = readFileSync(new URL("../../app/templates/index.html", import.meta.url), "utf8").replace(
        /<script[^>]*><\/script>/g,
        "",
    );
    const dom = new JSDOM(html, { url: "http://localhost:8000/", pretendToBeVisual: true });
    const { window } = dom;
    window.HTMLElement.prototype.scrollIntoView = function scrollIntoView(options) {
        this.scrolledWith = options;
    };
    window.scrollTo = (options) => {
        window.scrolledTo = options;
    };
    window.HTMLCanvasElement.prototype.getContext = () => null; // jsdom cannot draw; the code copes with that
    window.anchorClicks = [];
    window.HTMLAnchorElement.prototype.click = function click() {
        window.anchorClicks.push({ href: this.href, download: this.download });
    };
    globalThis.window = window;
    globalThis.document = window.document;
    return dom;
}

/** An element's natural size is 0 in jsdom (nothing is loaded); give it one. */
export function giveSize(element, { natural = [800, 600], client = [400, 300], offset = [0, 0] } = {}) {
    const define = (name, value) => Object.defineProperty(element, name, { value, configurable: true });
    define("naturalWidth", natural[0]);
    define("naturalHeight", natural[1]);
    define("clientWidth", client[0]);
    define("clientHeight", client[1]);
    define("offsetLeft", offset[0]);
    define("offsetTop", offset[1]);
    define("complete", true);
}

export function jsonResponse(body, { status = 200, headers = {} } = {}) {
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
}

/** A stand-in for EventSource that the test drives by hand. */
export class FakeEventSource {
    static last = null;

    constructor(url) {
        this.url = url;
        this.listeners = {};
        this.closed = false;
        this.onerror = null;
        FakeEventSource.last = this;
    }

    addEventListener(name, handler) {
        this.listeners[name] = handler;
    }

    emit(name, data) {
        this.listeners[name]({ data: JSON.stringify(data) });
    }

    close() {
        this.closed = true;
    }
}
