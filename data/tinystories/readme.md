# TinyStories data preparation

Create the fixed debugging subset:

```bash
uv run python data/tinystories/prepare.py --mode debug
```

This streams the first 10,000 `train` stories and first 1,000 `validation`
stories from the pinned dataset revision. It writes nanoGPT-compatible files to
`data/tinystories_debug/`. Binary data and `meta.pkl` stay local; commit the
generated `manifest.json` so the source revision, token counts, and SHA256 hashes
remain reviewable.

After the debugging pipeline passes, prepare both complete splits with:

```bash
uv run python data/tinystories/prepare.py --mode full
```

Complete files are written to `data/tinystories_full/`. The script streams and
writes incrementally instead of holding the dataset or all tokens in memory.
