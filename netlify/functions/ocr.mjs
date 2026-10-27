const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" }
});

export default async request => {
  if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);

  const serviceUrl = process.env.OCR_SERVICE_URL?.replace(/\/+$/, "");
  const token = process.env.OCR_SERVICE_TOKEN;
  if (!serviceUrl || !token) {
    return json({ error: "Image OCR is not configured. Set OCR_SERVICE_URL and OCR_SERVICE_TOKEN in Netlify." }, 503);
  }

  let form;
  try { form = await request.formData(); }
  catch { return json({ error: "Could not read the uploaded image." }, 400); }

  const image = form.get("image");
  const mode = form.get("mode") === "clean" ? "clean" : "photo";
  if (!image || typeof image.arrayBuffer !== "function") return json({ error: "Choose an image to scan." }, 400);
  if (image.size > 4 * 1024 * 1024) return json({ error: "Image must be 4 MB or smaller." }, 413);

  const forwarded = new FormData();
  forwarded.append("image", new Blob([await image.arrayBuffer()], { type: image.type || "application/octet-stream" }), image.name || "upload");
  forwarded.append("mode", mode);

  try {
    const upstream = await fetch(`${serviceUrl}/ocr`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: forwarded
    });
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" }
    });
  } catch (error) {
    console.error("OCR service request failed", error);
    return json({ error: "The OCR service is unavailable. Try again in a moment." }, 502);
  }
};
