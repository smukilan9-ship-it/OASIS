"""
send_chat — REMOVED from the shipped application (2026-07-25).

This was an LLM "ask questions about your results" helper on `oasis.webui.api.API`. It was
removed for three reasons:

1. **It was already dead code.** No element in `oasis/webui/index.html` ever called
   `window.pywebview.api.send_chat`; the chat panel it was written for was never built.
2. **It cannot ship in a standalone executable.** It requires network access plus a user-supplied
   `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`, and pulled `google-genai` + `anthropic` (and their
   transitive HTTP/auth stacks) into the bundle for a feature nobody could reach.
3. **It sends analysis results to a third party.** OASIS is used on clinical tissue images. A
   feature that transmits per-image cell counts and positivity off-machine needs an explicit,
   informed consent flow, not an incidental API method.

If an LLM narrative is wanted later, it should be reintroduced deliberately: opt-in, with a
visible statement of exactly what leaves the machine, and with the pipeline's own deterministic
summary as the default. Note that the system prompt below hardcodes stale method text
("InstanSeg brightfield_nuclei, DAB threshold 0.2 OD") rather than reading the run's actual
provenance — a reimplementation should use the recorded values.

Retained verbatim for reference. Requires: google-genai, anthropic, python-dotenv.
"""
import json


def send_chat(self, message, context):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        setup    = self.get_setup()
        provider = setup.get("ai_provider", "gemini")
        model    = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-4-20250514"
        metrics  = context.get("metrics", [])
        total    = sum(m.get("total_cells", 0) for m in metrics)
        pos      = sum(m.get("positive", 0) for m in metrics)
        avg      = pos / total * 100 if total > 0 else 0
        system = f"""You are an expert IHC analysis assistant.
Results: {len(metrics)} images, {total:,} total cells, {pos:,} positive, {avg:.2f}% avg positivity.
Method: InstanSeg brightfield_nuclei, DAB threshold 0.2 OD.
Per image: {json.dumps([{'name': m['name'], 'cells': m['total_cells'], 'positivity': m['positivity']} for m in metrics])}
Summary: {context.get('summary', '')}
Answer concisely and scientifically. Methods sections use past tense passive voice."""
        full = f"{system}\n\nUser: {message}"
        if provider == "gemini":
            from google import genai
            import os as _os
            client = genai.Client(api_key=_os.getenv("GEMINI_API_KEY"))
            r = client.models.generate_content(model=model, contents=full)
            return {"ok": True, "response": r.text.strip()}
        else:
            import anthropic, os as _os
            client = anthropic.Anthropic(api_key=_os.getenv("ANTHROPIC_API_KEY"))
            r = client.messages.create(model=model, max_tokens=800,
                                       messages=[{"role": "user", "content": full}])
            return {"ok": True, "response": r.content[0].text.strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
