# Contributing to OASIS

Thanks for your interest. This document covers the three things you are most
likely to want: **reporting a problem**, **getting help**, and **contributing a
change**.

## Reporting issues

Open an issue at
<https://github.com/smukilan9-ship-it/OASIS/issues>. The tracker is
public and readable without a GitHub account.

For anything that produces a wrong or surprising *number*, please include:

- the OASIS version or commit SHA,
- your OS and Python version,
- the `*_summary.json` for the affected run — it records the pixel size, DAB
  threshold, segmenter, device, and git SHA, which is usually enough to
  reproduce the run,
- what you expected versus what you got.

**Please do not attach patient images or any identifiable data.** A crop of
non-identifiable tissue, or a synthetic image that reproduces the behaviour, is
enough. If a problem can only be shown with restricted data, say so in the issue
and describe it instead.

## Seeking support

- **Usage questions** ("how do I calibrate the scale bar?", "which null model
  applies here?") — open a **Question** issue. There is no separate forum or
  mailing list; questions in the tracker help the next person with the same one.
- **Interpretation of results** — please include the run's summary JSON and, if
  the run was spatial, the registration certification verdict. Most "the number
  looks wrong" reports turn out to be a registration or bandwidth gate doing its
  job, and the verdict says which.

Expect a response within about a week. This is research software maintained
alongside other work, not a commercial product.

## Contributing changes

1. Open an issue first for anything beyond a typo, so the approach can be agreed
   before you spend time on it.
2. Fork, branch from `main`, and keep the change focused.
3. Run the test suite: `python -m pytest`. It must pass.
4. If you touch anything that affects a reported number, see the next section.
5. Open a pull request describing *what changed and why*, and what you ran to
   convince yourself it is right.

### Changes that affect scientific output

This is the part that differs from a typical project. OASIS reports statistics
that people may act on, so a change to segmentation, deconvolution, registration,
null models, or thresholds is not accepted on "the tests pass" alone.

- Pinned dependencies in `requirements.txt` are pinned deliberately. numpy,
  scipy, and opencv version drift can move percentiles, KDE binning, and Otsu
  output, which can flip a significance verdict. Do not relax `==` to `>=`.
- Add or update a validation in `validation/registry.py` covering the behaviour,
  and include the resulting report bundle's key metrics in the PR.
- If a change alters existing numbers, say so explicitly and quantify the change.
  A silent shift in output is the failure mode this project most wants to avoid.
- Prefer failing closed. If a result cannot be justified, the pipeline should
  refuse to report it rather than report it with a caveat.

### Running the validations

```bash
python -m validation.run --list          # what exists, and which datasets it needs
python -m validation.run all --tier instant   # everything that needs no dataset
```

Validations requiring datasets skip with a message naming the dataset and its
source. Datasets are never committed; see `validation/datasets/README.md`.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
