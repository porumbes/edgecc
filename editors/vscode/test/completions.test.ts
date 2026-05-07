import * as assert from "assert";
import {
  COMPUTE_OPS,
  MERGE_OPS,
  TYPES,
  EMPTY_VALUES,
} from "../src/completions";
import { detectContext } from "../src/completions";

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

describe("detectContext", () => {
  it("returns 'compute-op' right after map(", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map("), "compute-op");
  });

  it("returns 'compute-op' right after reduce(", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map(mul) reduce("), "compute-op");
  });

  it("returns 'merge-op' after a comma in map(...)", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map(mul, "), "merge-op");
  });

  it("returns 'merge-op' after a comma in reduce(...)", () => {
    assert.strictEqual(
      detectContext("Z[m] = A . B :: map(mul, intersect) reduce(min, "),
      "merge-op",
    );
  });

  it("returns 'type' after `:` in a tensor decl", () => {
    assert.strictEqual(detectContext("tensor X[R=K] : "), "type");
    assert.strictEqual(detectContext("tensor X[R=K, C=M] : "), "type");
  });

  it("returns 'empty-value' after `empty = `", () => {
    assert.strictEqual(detectContext("tensor X[R=K] : float, empty = "), "empty-value");
  });

  it("returns null inside a comment", () => {
    assert.strictEqual(detectContext("// map("), null);
    assert.strictEqual(detectContext("dim K = 3; // map(mul, "), null);
  });

  it("returns null inside a tensor rank list (no map/reduce in context)", () => {
    assert.strictEqual(detectContext("tensor X[R=K, "), null);
  });

  it("returns null on irrelevant prefixes", () => {
    assert.strictEqual(detectContext(""), null);
    assert.strictEqual(detectContext("dim K = "), null);
    assert.strictEqual(detectContext("init A = ["), null);
  });
});
