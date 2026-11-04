# V-JEPA2.1-AI4Eyes

V-JEPA 2.1-based models for AI4Eyes clinical signs.

- **[`video/`](video/README.md)** — three video-based temporal action
  detection tests: NITBUT, Invasive-TBUT, Blink Mechanics. Start there.
- **`image/`** — coming later: corneal staining, hyperemia, tear meniscus,
  meibography, lid margin. Will share `video/`'s manifest schema so the
  two can be pretrained jointly once both exist.

No patient data, checkpoints, or credentials live in this repository (see
`.gitignore`) — every data/model path is supplied via environment
variables, documented in [`video/README.md`](video/README.md#configuration).
