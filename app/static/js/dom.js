// Tiny DOM helpers. All text is inserted with textContent, never as HTML.

export const $ = (id) => document.getElementById(id);

export const show = (id) => $(id).classList.remove("hidden");
export const hide = (id) => $(id).classList.add("hidden");

/** el("div", {class: "card", text: "Hello", attrs: {role: "group"}}, child1, child2) */
export function el(tag, options = {}, ...children) {
    const node = document.createElement(tag);
    if (options.class) {
        node.className = options.class;
    }
    if (options.text !== undefined) {
        node.textContent = options.text;
    }
    if (options.id) {
        node.id = options.id;
    }
    for (const [name, value] of Object.entries(options.attrs ?? {})) {
        node.setAttribute(name, value);
    }
    for (const child of children) {
        if (child) {
            node.appendChild(child);
        }
    }
    return node;
}

export const setText = (id, value, fallback = "-") => {
    $(id).textContent = value === null || value === undefined || value === "" ? fallback : String(value);
};

export const chip = (text, tone = "") => el("span", { class: `chip ${tone}`.trim(), text });
