import js from "@eslint/js";
import globals from "globals";

export default [
    js.configs.recommended,
    {
        files: ["app/static/js/**/*.js"],
        languageOptions: { ecmaVersion: 2023, sourceType: "module", globals: globals.browser },
    },
    {
        files: ["tests/js/**/*.js", "eslint.config.js"],
        languageOptions: { ecmaVersion: 2023, sourceType: "module", globals: { ...globals.node, ...globals.browser } },
    },
    {
        rules: {
            eqeqeq: "error",
            "no-var": "error",
            "prefer-const": "error",
            "no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
            complexity: ["error", 10],
            "max-depth": ["error", 3],
            "max-params": ["error", 5],
            "no-shadow": "error",
            "no-nested-ternary": "error",
            "no-param-reassign": "error",
            "prefer-template": "error",
            curly: "error",
        },
    },
];
