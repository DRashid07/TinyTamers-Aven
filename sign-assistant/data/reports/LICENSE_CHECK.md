# Licence check: AzSLD

Owner: A (Data/ML). Task P1, checked on 2026-10-09.

## Where the licence text comes from

The team's own copy of the licence text was **not provided**: the task prompt still contained the
placeholder `<PASTE LICENCE TEXT>`. This check therefore uses the Zenodo record, fetched by Claude Code
on 2026-10-09 from `https://zenodo.org/api/records/14222948`:

- DOI 10.5281/zenodo.14222948 (concept DOI 10.5281/zenodo.13627300), title "AzSLD - Azerbaijani Sign Language Dataset",
  creators Alishzade, Nigar and Hasanov, Jamaladdin.
- `metadata.license.id`: `cc-by-4.0`
- `metadata.access_right`: `open`
- Description, "Accessibility": "The AzSLD is available under Creative Commons Attribution 4.0 International
  with free access for academic research through Zenodo."
- Description, "Ethical Transparency": "All participants provided informed consent for collecting, publishing,
  and using the data, ensuring compliance with ethical research standards."

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/legalcode

**To do:** one team member opens https://doi.org/10.5281/zenodo.14222948 and confirms the licence shown there.

## What it allows for our non-commercial hackathon demo

CC BY 4.0 lets anyone copy, redistribute and adapt the material for any purpose, on condition that they give
credit, link the licence, say if changes were made, and add no further restrictions. For us:

- Extracting landmarks and training a model on the videos: allowed (adaptation).
- Showing word clips on the text-to-signs page: allowed, with the attribution below and a note that the
  clips were cut and re-encoded.
- Keeping raw videos, clips, landmarks and weights out of git is a team rule, not a licence requirement.

## Ambiguity

The phrase "with free access for academic research" could be read as limiting use to academic research.
The licence itself (CC BY 4.0) has no such limit, and a hackathon demo is non-commercial anyway, so we
treat our use as permitted. Ask the authors (slr.project.ada@gmail.com) before any commercial use.

The licence covers copyright only. The clips show the signers' faces; the record says they consented to
publishing and use. We do not try to identify signers.

## Attribution we must show

In README.md, in the demo UI next to the clips, and on any slides:

> Sign videos: AzSLD – Azerbaijani Sign Language Dataset, N. Alishzade and J. Hasanov, CC BY 4.0,
> DOI 10.5281/zenodo.14222948. Clips cut and re-encoded by the team.
> Paper: Alishzade, N., Hasanov, J. (2025). AzSLD: Azerbaijani sign language dataset for fingerspelling,
> word, and sentence translation with baseline software. Data in Brief 58, 111230.
> https://doi.org/10.1016/j.dib.2024.111230

## Status

**OK for the non-commercial hackathon demo with the attribution above**, provided a team member confirms
the licence on the Zenodo page (the team's own copy of the text is missing).
