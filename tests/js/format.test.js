import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
    ageDetail,
    boxStyle,
    capitalise,
    downloadName,
    genderDetail,
    lowerWords,
    megabytes,
    orDash,
    percent,
    VERDICTS,
    verdictFor,
} from "../../app/static/js/format.js";

describe("format", () => {
    it("words every outcome the API can return", () => {
        assert.deepEqual(verdictFor("PASSED"), ["Passed all checks", "success"]);
        assert.deepEqual(verdictFor("POSSIBLE_MINOR"), ["Age check needed (possible minor)", "warning"]);
        assert.deepEqual(verdictFor("SOMETHING_NEW"), ["SOMETHING_NEW", "neutral"]);
        assert.equal(Object.keys(VERDICTS).length, 10);
    });

    it("formats shares, sizes and missing values", () => {
        assert.equal(percent(0.873), "87%");
        assert.equal(percent(0.873, 1), "87.3%");
        assert.equal(percent(null), null);
        assert.equal(percent(undefined), null);
        assert.equal(orDash(null), "-");
        assert.equal(orDash(""), "-");
        assert.equal(orDash(0), "0");
        assert.equal(megabytes(5 * 1024 * 1024), "5.00");
        assert.equal(lowerWords("PARTIAL_OR_CROPPED"), "partial or cropped");
        assert.equal(capitalise("running"), "Running");
    });

    it("describes the age of an estimated face and of an unsure one", () => {
        const estimated = {
            age_status: "Estimated",
            estimated_age: 34,
            age_range: [26, 42],
            age_group_probability: 0.19,
            notes: [],
        };
        assert.equal(ageDetail(estimated), "Estimated 34 years, 80% range 26 to 42. Probability of this bracket: 19%.");
        assert.equal(ageDetail({ age_status: "Unsure", notes: ["Views disagree.", "Be careful."] }), "Views disagree. Be careful.");
        assert.equal(ageDetail({ age_status: "Unsure", notes: [] }), "The models could not support an age.");
    });

    it("describes the gender score", () => {
        assert.equal(genderDetail({ gender_confidence: 0.971 }), "Model score 97.1%");
        assert.equal(genderDetail({ gender_confidence: null }), "");
    });

    it("places a percent box inside a sized element", () => {
        assert.deepEqual(boxStyle([25, 50, 75, 100], 400, 200), { left: 100, top: 100, width: 200, height: 100 });
        assert.deepEqual(boxStyle([0, 0, 50, 50], 200, 200, 10, 20), { left: 10, top: 20, width: 100, height: 100 });
    });

    it("names downloads", () => {
        assert.equal(downloadName("report", "json", 123), "report-123.json");
        assert.match(downloadName("report", "json"), /^report-\d+\.json$/);
    });
});
