# Technical report

LaTeX report for our FREUID Challenge 2026 reproducibility package (required
for prize eligibility — see the top-level `README.md` and
[freuid2026.microblink.com/reproducibility.html](https://freuid2026.microblink.com/reproducibility.html)).

## Files

| File | Purpose |
| ---- | ------- |
| `freuid_technical_report.tex` | Main report (method, data, inference, results, reproducibility) |
| `references.bib` | Bibliography (challenge citation + ConvNeXt-V2 + timm) |

## Build

```bash
latexmk -pdf freuid_technical_report.tex
# or:
pdflatex freuid_technical_report.tex && bibtex freuid_technical_report && pdflatex freuid_technical_report.tex && pdflatex freuid_technical_report.tex
```

No LaTeX toolchain was available in the development sandbox this report was
drafted in — compile it (locally, via Overleaf, or in CI) before the July 15
deadline and commit/publish the resulting PDF alongside (or linked from) this
repository.

## Outstanding `[TODO: ...]` placeholders

Search the `.tex` file for `[TODO` to find every place that still needs
team-specific info or a result that is only knowable after the private test
images are released on July 13, 2026:

- Team name, author names/affiliations, contact email, repo URL, commit SHA.
- The validated outcome of the in-flight captured-image-oversampling
  experiment (Section "FREUID training set").
- Private leaderboard score (Section "Results").
- CPU-only and private-set wall-clock timing (Sections "Inference" and
  "Reproducibility").
- Team/contributions and acknowledgments sections.
