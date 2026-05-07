import * as assert from "assert";
import {
  COMPUTE_OPS,
  MERGE_OPS,
  TYPES,
  EMPTY_VALUES,
} from "../src/completions";

describe("completion item lists", () => {
  it("COMPUTE_OPS contains all 7 built-in compute operators", () => {
    const labels = COMPUTE_OPS.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["add", "and", "max", "min", "mul", "or", "sub"]);
  });

  it("MERGE_OPS contains all 9 supported merge operators", () => {
    const labels = MERGE_OPS.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, [
      "intersect", "nopass", "passthrough", "take_left", "take_left_only",
      "take_right", "take_right_only", "union", "xor",
    ]);
  });

  it("TYPES contains exactly integer, float, boolean", () => {
    const labels = TYPES.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["boolean", "float", "integer"]);
  });

  it("EMPTY_VALUES contains exactly 0, 0.0, inf, true, false", () => {
    const labels = EMPTY_VALUES.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["0", "0.0", "false", "inf", "true"]);
  });

  it("every MERGE_OP has documentation containing a markdown table", () => {
    for (const op of MERGE_OPS) {
      assert.ok(
        op.documentation && op.documentation.includes("| A | B | Z |"),
        `merge op '${op.label}' missing truth-table documentation`,
      );
    }
  });

  it("every item has a non-empty detail string", () => {
    for (const list of [COMPUTE_OPS, MERGE_OPS, TYPES, EMPTY_VALUES]) {
      for (const item of list) {
        assert.ok(
          item.detail && item.detail.length > 0,
          `item '${item.label}' missing detail`,
        );
      }
    }
  });
});
