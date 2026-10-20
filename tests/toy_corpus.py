"""A tiny corpus shared by tests that need real retrievers but not a real dataset."""

CORPUS = {
    "d1": {"title": "Cats", "text": "A cat sat on the mat."},
    "d2": {"title": "Dogs", "text": "Dogs chase cats around the garden."},
    "d3": {"title": "Foxes", "text": "The quick brown fox jumps over the lazy dog."},
    "d4": {"title": "Vaccines", "text": "Vaccines train the immune system."},
}
QUERIES = ["cat on a mat", "dog chasing", "immune response", "the"]
