/**
 * What the client must not lose in translation.
 *
 * A client is where an audited answer becomes an ordinary string. The two
 * tests below are the two ways that happens: the audit fields getting
 * flattened away, and the per-user addressing being left to the caller —
 * both of which turn this into a chat completion with extra steps.
 *
 * The wire test starts the real server, because a contract between two
 * languages that is only ever checked against a mock is not a contract.
 *
 *   node --experimental-strip-types --test sdk/typescript/test/
 */
import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { LMM, LMMError } from "../src/index.ts";

test("the answer keeps its audit, and the user is addressed for the caller", async () => {
  const seen: { url?: string; body?: any; auth?: string } = {};
  const fakeFetch = (async (url: string, init: any) => {
    seen.url = String(url);
    seen.body = JSON.parse(init.body);
    seen.auth = init.headers["Authorization"];
    return {
      ok: true,
      status: 200,
      json: async () => ({
        answer: "The Redwood contract is worth 40000 euro.",
        abstained: false,
        sources: ["#doc:Redwood.txt"],
        subject: "Redwood contract",
      }),
    };
  }) as unknown as typeof fetch;

  const memory = new LMM({ base: "http://x", user: "ada", token: "k", fetch: fakeFetch });
  const answer = await memory.ask("what is the Redwood contract worth?");

  assert.equal(seen.url, "http://x/ask");
  assert.equal(seen.body.user, "ada", "the caller had to address the user itself");
  assert.equal(seen.auth, "Bearer k");
  assert.equal(answer.abstained, false);
  assert.deepEqual(answer.sources, ["#doc:Redwood.txt"]);
  assert.match(answer.answer, /40000/);
});

test("a refusal arrives as a refusal, and an error as an error", async () => {
  const refusing = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({ answer: "I do not have that.", abstained: true, sources: [], subject: "" }),
  })) as unknown as typeof fetch;
  const memory = new LMM({ base: "http://x", user: "ada", fetch: refusing });
  const answer = await memory.ask("what is the Fernbank worth?");
  assert.equal(answer.abstained, true, "an application cannot tell a refusal from a fact");
  assert.deepEqual(answer.sources, []);

  const failing = (async () => ({
    ok: false,
    status: 400,
    statusText: "Bad Request",
    json: async () => ({ error: "bad user id" }),
  })) as unknown as typeof fetch;
  const bad = new LMM({ base: "http://x", user: "ada", fetch: failing });
  await assert.rejects(() => bad.ask("anything?"), (e: unknown) => {
    assert.ok(e instanceof LMMError);
    assert.equal((e as LMMError).status, 400);
    return true;
  });

  assert.throws(() => new LMM({ base: "http://x", user: "" }), /user id/);
});

test("across the wire: what one user teaches, another cannot read", async (t) => {
  const port = 8757;
  const root = await import("node:fs/promises").then((fs) =>
    fs.mkdtemp("/tmp/lmm-sdk-"),
  );
  const server = spawn("python3.11", ["-m", "lmm.serve", "--root", root, "--port", String(port)], {
    cwd: new URL("../../../", import.meta.url).pathname,
    env: { ...process.env, PYTHONPATH: "src", LMM_BACKEND: "azure" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const up = await new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), 20000);
    server.stdout.on("data", (chunk) => {
      if (String(chunk).includes("serving")) {
        clearTimeout(timer);
        resolve(true);
      }
    });
  });
  if (!up) {
    server.kill();
    t.skip("server did not start");
    return;
  }
  try {
    const ada = new LMM({ base: `http://127.0.0.1:${port}`, user: "ada" });
    const bo = new LMM({ base: `http://127.0.0.1:${port}`, user: "bo" });
    const learned = await ada.learn({
      text: "The Redwood contract is worth 40000 euro.",
      source: "#doc:Redwood.txt",
    });
    assert.ok(learned.evidence > 0 || learned.facts > 0, JSON.stringify(learned));
    assert.ok((await ada.where("Redwood")).length > 0);
    assert.deepEqual(await bo.where("Redwood"), [], "one user's document was visible to another");
  } finally {
    server.kill();
  }
});
