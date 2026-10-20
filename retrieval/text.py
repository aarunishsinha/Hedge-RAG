def doc_text(doc: dict) -> str:
    """Text a retriever indexes for one doc: title + " " + text (BEIR convention)."""
    title = doc.get("title", "")
    text = doc.get("text", "")
    return f"{title} {text}".strip()
