
// Vercel serverless function: calls the Hugging Face Space (Gradio API).
// Env vars: SPACE_URL (required), HF_TOKEN (only if the Space is private)
export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }
  const { prompt } = req.body || {};
  if (!prompt || typeof prompt !== "string" || prompt.length > 1000) {
    return res.status(400).json({ error: "Provide a prompt (max 1000 characters)." });
  }
  let base = process.env.SPACE_URL;
  if (!base) return res.status(500).json({ error: "SPACE_URL is not configured." });
  if (base.endsWith("/")) base = base.slice(0, -1);

  const headers = { "Content-Type": "application/json" };
  if (process.env.HF_TOKEN) headers.Authorization = "Bearer " + process.env.HF_TOKEN;

  try {
    const start = await fetch(base + "/gradio_api/call/generate", {
      method: "POST", headers, body: JSON.stringify({ data: [prompt] }),
    });
    if (!start.ok) return res.status(502).json({ error: "Space returned " + start.status + " (it may be waking up, retry in a minute)." });
    const { event_id } = await start.json();

    const result = await fetch(base + "/gradio_api/call/generate/" + event_id, { headers });
    const body = await result.text();

    let event = "", answer = null;
    for (const line of body.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:") && event === "complete") answer = JSON.parse(line.slice(5).trim())[0];
    }
    if (answer === null) return res.status(502).json({ error: "The model returned no answer." });
    return res.status(200).json({ response: answer });
  } catch (e) {
    return res.status(502).json({ error: "Could not reach the Hugging Face Space." });
  }
}
