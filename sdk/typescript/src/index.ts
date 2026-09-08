/**
 * The TypeScript side of the same door.
 *
 * This client exists so that the thing an application receives from LMM in
 * a browser or a Node service is the SAME OBJECT a Python caller receives:
 * an answer that knows whether it was abstained and which documents it
 * rests on. A client that returned a bare string would quietly discard the
 * only property that distinguishes this memory from a chat completion, and
 * every application built on it would have to re-derive "did it actually
 * know that?" from prose.
 *
 * No dependencies, and no runtime of its own: `fetch` and the endpoints of
 * `lmm.serve`.
 *
 *   const m = new LMM({ base: "http://localhost:8000", user: "ada" });
 *   await m.learn({ text: "Ada leads R&D.", source: "#doc:team.txt" });
 *   const a = await m.ask("who leads R&D?");
 *   if (a.abstained) console.log("memory does not hold this");
 *   else console.log(a.answer, a.sources);
 */

export interface Answer {
  /** What the memory said — including the honest refusal. */
  answer: string;
  /** True when the memory declined rather than answered. Never guess past this. */
  abstained: boolean;
  /** The documents the claim rests on, as the store stamped them. */
  sources: string[];
  subject: string;
}

export interface Learned {
  facts: number;
  evidence: number;
  source: string;
  adapter: string;
  warnings: string[];
}

export interface Located {
  source: string;
  hits: number;
}

export interface Options {
  /** Where lmm.serve is listening. */
  base?: string;
  /** Whose memory. Every user's memory is their own, server-side. */
  user: string;
  /** Sent as `Authorization: Bearer …` when the server sets LMM_TOKEN. */
  token?: string;
  fetch?: typeof fetch;
}

export class LMMError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "LMMError";
    this.status = status;
  }
}

export class LMM {
  private readonly base: string;
  private readonly user: string;
  private readonly token?: string;
  private readonly doFetch: typeof fetch;

  constructor(options: Options) {
    this.base = (options.base ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
    this.user = options.user;
    this.token = options.token;
    this.doFetch = options.fetch ?? globalThis.fetch;
    if (!this.user) throw new Error("a user id is required: memories are per user");
  }

  private async post<T>(route: string, body: Record<string, unknown>): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    const response = await this.doFetch(`${this.base}${route}`, {
      method: "POST",
      headers,
      body: JSON.stringify({ user: this.user, ...body }),
    });
    const payload = (await response.json()) as Record<string, unknown>;
    if (!response.ok) {
      throw new LMMError(response.status, String(payload.error ?? response.statusText));
    }
    return payload as T;
  }

  /** Teach the memory some text, or a file path the server can read. */
  learn(what: { text?: string; path?: string; source?: string; deep?: boolean }): Promise<Learned> {
    return this.post<Learned>("/learn", what as Record<string, unknown>);
  }

  /**
   * Ask. The result carries its own audit: check `abstained` before you
   * put the text in front of a person as a fact.
   */
  ask(question: string, options: { fluent?: boolean } = {}): Promise<Answer> {
    return this.post<Answer>("/ask", { question, ...options });
  }

  /** A sourced document written from what the memory holds. */
  compose(brief: string): Promise<{ text: string; sources: string[] }> {
    return this.post("/compose", { brief });
  }

  /** Which documents mention a term, and how often. */
  async where(term: string): Promise<Located[]> {
    const got = await this.post<{ where: Located[] }>("/where", { term });
    return got.where;
  }
}

export default LMM;
