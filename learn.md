# Learn: OASIS explained in plain English

This document is a companion to the presentation
`outputs/ihc_analysis_47page_cleaned_canva_safe.pptx`, using `ihc.md` as the
technical source of truth. It follows the same flow as the PPT:

1. Quantification: find cells and count marker-positive cells.
2. Why serial-section co-expression is not defensible.
3. Spatial association: the honest alternative for serial sections.
4. Restaining: when true same-cell co-expression becomes possible.
5. Validation: what was tested, what worked, and what is still limited.

It is written for someone who does not already know pathology, microscopy,
statistics, or this codebase. The goal is not to hide the technical details. The
goal is to explain them slowly enough that the details make sense.

---

## 1. The project in one sentence

The project analyzes stained tissue images so that a computer can answer careful
questions such as:

> "Where are the marker-positive cells, and do two marker-positive cell
> populations tend to appear near each other more than expected by chance?"

The most important word is **careful**. The software is deliberately conservative.
It tries not to make a stronger biological claim than the images can support.

The project has three related but different analysis ideas:

| Part of the project | Plain-English goal | What it can say |
|---|---|---|
| Quantification | Count cells and decide which ones are marker-positive in one image. | "This image has N cells, M of them positive for this stain at this threshold." |
| Spatial association | Compare two cell populations from serial sections. | "These two populations are near each other in tissue more than expected under a calibrated null model." |
| Restaining / same-section analysis | Measure two markers on the same physical tissue section. | "These exact segmented cells are A-only, B-only, double-positive, or neither," if alignment and thresholds are valid. |

That distinction matters because the scientific meaning changes depending on how
the images were produced.

---

## 2. Slide 1: What immunohistochemistry is

**Immunohistochemistry**, usually shortened to **IHC**, is a way to make a
specific protein visible in tissue.

Imagine a tissue image as a city seen from above. The cells are buildings, but
you cannot tell which buildings contain a specific person. IHC is like sending a
highly selective "spotter" into the city. The spotter is an **antibody**. It
sticks only to a specific target protein, such as CD8. Then a chemical reaction
adds a visible color where that target is found.

In this project the images are usually **H-DAB brightfield IHC**:

- **H** means **hematoxylin**, the blue/purple stain that shows cell nuclei.
- **DAB** is the brown stain that marks the protein of interest.
- **Brightfield** means this is ordinary microscope-style color imaging, not
  fluorescent imaging.

So the computer mainly uses two visual clues:

- Blue/purple nuclei tell it where cells are.
- Brown stain tells it whether a marker is present.

The rest of the project turns those colors into measurements, and then turns the
measurements into honest biological statements.

---

## 3. Slides 2-7: Quantification workflow

Quantification is the basic counting pipeline. It answers:

> "In this image, how many cells did we find, and how many are positive for the
> selected marker?"

The PPT shows the workflow as eight steps.

### 3.1 Select input

The user can run one image or a whole folder. A single image is useful for
checking a setting. A folder batch is useful when the same settings should be
applied to many tiles or fields of view.

### 3.2 Assign stain

The software needs to know which marker each image represents. It can infer this
from the filename, for example `CD8`, `TIM-3`, or another marker token.

This is important because each marker can have a different threshold. A threshold
that is sensible for CD8 may not be sensible for TIM-3.

### 3.3 Resolve pixel size

A microscope image is made of pixels, but biology happens in physical distance:
micrometres, written **µm**. A micrometre is one thousandth of a millimetre.

The conversion is:

```text
physical distance in µm = distance in pixels × pixel size in µm/pixel
```

If the pixel size is wrong, every downstream distance is wrong:

- cell expansion radius,
- registration error,
- Ripley K radius,
- tissue area,
- spatial-association interpretation.

The software resolves pixel size in this order:

1. Per-image manual override.
2. TIFF or OME microscope metadata.
3. Session default entered in the UI.
4. Filename magnification mapping, such as `20x`.
5. Burned-in 100 µm scale-bar measurement.
6. Documented fallback of 0.5 µm/pixel.

**Important:** ordinary image DPI is not microscope calibration. DPI only says
how an image might print on paper. It does not reliably say how large a pixel is
in the tissue.

### 3.4 Segment nuclei

**Segmentation** means drawing boundaries around objects in the image. Here the
objects are nuclei or cell-like nuclear detections.

The software uses **QuPath** running headless, with **InstanSeg
`brightfield_nuclei`**, to find nuclei in the hematoxylin channel. In plain
language: a trained model looks for the blue/purple nuclear shapes and turns the
image into a list of detected objects.

For each detected object, the output stores:

- the cell/nucleus boundary,
- the centroid, meaning the center point,
- the stain measurements,
- the positive/negative classification.

### 3.5 Measure stain

After a cell object is found, the software measures how strong the brown stain is
inside the configured measurement region.

The measurement is an **optical density**, often abbreviated **OD**. In simple
terms:

- light color means low stain OD,
- darker brown means higher stain OD,
- higher OD means more marker signal.

The exact math behind OD is logarithmic:

```text
OD = -log10(observed light / background light)
```

You do not need to memorize the logarithm. The intuitive meaning is enough:
stronger stain gives a larger OD value.

### 3.6 Apply threshold

A **threshold** is the cutoff used to decide positive versus negative.

The current Quantification starting values in the PPT are:

| Marker | Starting OD threshold |
|---|---:|
| CD8 | 0.20 |
| TIM-3 | 0.10 |

The rule is:

```text
if measured OD ≥ threshold: call the cell positive
if measured OD < threshold: call the cell negative
```

These thresholds are **operational inputs**, not universal clinical truth. They
are starting settings used by the pipeline. Proper marker-label validation would
require expert-labelled positive and negative cells for the marker.

### 3.7 Classify

Each segmented object receives a label:

- positive, if it meets the stain threshold,
- negative, if it does not.

This classification depends on two things at once:

1. Did segmentation find the right object?
2. Did the stain measurement and threshold correctly classify it?

Those are separate validation questions. A segmentation validation can show that
the nuclei were found well, but it does not automatically prove that every marker
threshold is correct.

### 3.8 Export

The Quantification pipeline writes several files:

| Output | What it contains | Why it matters |
|---|---|---|
| CSV | Per-object table of measurements and labels. | Easy to inspect in Excel or scripts. |
| GeoJSON | Object geometries and measurements. | Lets the shapes be reopened or drawn later. |
| Summary JSON | Totals, thresholds, provenance. | Records exactly how the result was produced. |
| Overlay PNG | Boundaries drawn on the original image. | Visual quality control. |
| Dashboard / workbook | Human-readable summaries. | Easier review and reporting. |

The overlay is especially useful because it lets a human quickly ask:

> "Are the outlines actually sitting on the cells?"

---

## 4. Slides 4-7: Segmentation validation and what it does not prove

The PPT separates **segmentation validation** from **marker-label validation**.
That distinction is very important.

### 4.1 What segmentation validation tests

Segmentation validation compares the computer's detected nuclei against an expert
segmentation mask.

It answers:

> "Did the computer find the nuclei in roughly the right places?"

It does **not** answer:

> "Did the computer correctly decide which cells are CD8-positive or TIM-3-positive?"

Those are different tasks.

### 4.2 The basic counting terms

When comparing computer segmentation to expert segmentation, every object or
pixel can fall into one of these categories:

| Term | Meaning |
|---|---|
| TP, true positive | The expert says there is a nucleus, and the computer found it. |
| FP, false positive | The computer found a nucleus, but the expert did not mark one there. |
| FN, false negative | The expert marked a nucleus, but the computer missed it. |
| TN, true negative | Both agree that background is background. |

For segmentation, **true negatives are usually not very informative**, because
most of an image is background. A method can look artificially good if it gets
background right but misses many nuclei. That is why precision, recall, F1, Dice,
and IoU are more useful than plain pixel accuracy.

### 4.3 Precision

```text
precision = TP / (TP + FP)
```

Plain meaning:

> "Of everything the computer called a nucleus, what fraction was actually a
> nucleus?"

High precision means the computer is not hallucinating many extra nuclei.

Low precision means it is drawing too many false objects.

### 4.4 Recall

```text
recall = TP / (TP + FN)
```

Plain meaning:

> "Of all expert-marked nuclei, what fraction did the computer find?"

High recall means the computer is not missing many nuclei.

Low recall means it leaves many real nuclei undetected.

### 4.5 F1 score

```text
F1 = 2 × precision × recall / (precision + recall)
```

Plain meaning:

> "A single score that rewards precision and recall together."

F1 is low if either precision or recall is low. That is why it is useful when we
care about both false detections and missed detections.

### 4.6 Dice score

For masks, Dice is:

```text
Dice = 2 × overlap area / (computer area + expert area)
```

Plain meaning:

> "How much do the two masks overlap, while giving credit to both sides?"

Dice ranges from 0 to 1:

- 1 means perfect overlap,
- 0 means no overlap.

Dice is often used for segmentation masks.

### 4.7 IoU, also called Jaccard index

```text
IoU = overlap area / union area
```

The **union** is everything covered by either mask.

Plain meaning:

> "Out of all pixels either side marked as object, how many did both agree on?"

IoU is stricter than Dice. For the same prediction, IoU is usually lower.

### 4.8 Micro-F1

The PPT reports a full external audit micro-F1 of **0.776** across 268 HNSCC
tiles.

**Micro-F1** means we pool all true positives, false positives, and false
negatives across the dataset first, then calculate one F1 score.

Plain meaning:

> "Treat the whole dataset like one big image-set and ask how the detector did
> overall."

This is different from taking the average of each tile's F1. Micro-F1 gives more
weight to tiles with more nuclei.

### 4.9 What the HNSCC audit showed

The PPT's full external audit reports:

- 268 expert nuclear masks,
- 95,519 reference nuclei,
- 85,336 predicted nuclei,
- micro precision 0.822,
- micro recall 0.734,
- micro-F1 0.776,
- median tile F1 0.786,
- F1 range 0.178 to 0.919.

Layman interpretation:

> The segmentation pipeline works and is auditable, but it is not perfect. It
> finds many nuclei correctly, misses some, and adds some extras. Performance
> varies by tile.

The audit also found that image correspondence matters. Some public tiles were
not perfectly aligned between the mIF reference and the mIHC hematoxylin image.
So low F1 is not always purely an InstanSeg failure; sometimes the "expert mask"
and target image are not locally in the same place.

---

## 5. Slide 6: TIM-3 membrane remeasurement, what was tried, and why it was disabled

TIM-3 is biologically membrane-associated. That means the signal can sit around
the cell edge, not in the nucleus.

The project tried a membrane/cytoplasm-ring measurement:

1. Start with the segmented nucleus.
2. Expand outward by a small physical distance.
3. Clip the expanded region so it does not cross into neighboring cells.
4. Measure stain in that ring.

The clipping uses a **Voronoi** idea. Imagine each nucleus owns the territory
closer to itself than to any other nucleus. The ring cannot cross the border into
a neighbor's territory. That prevents one cell from stealing another cell's
membrane stain.

The problem was not the geometry. The problem was calibration.

In the controlled TIM-3 ring experiment:

- original QuPath-positive objects: **61**,
- direct ring-threshold reclassification: **4,433**,
- resulting positive fraction: **32.0%**.

That jump was too large to trust. The same threshold was being reused on a
different measurement distribution. Also, a thin membrane signal can be diluted
when averaged over a whole ring.

Therefore the visible workflow disables TIM-3 ring reclassification until a
proper manual validation set exists.

Plain meaning:

> The project tried a biologically motivated measurement, found that it was not
> calibrated, and did not hide that failure. The safer current choice is to keep
> the original classification rather than ship a flashy but unreliable TIM-3
> membrane call.

---

## 6. Slides 8-12: The tempting question, and why serial sections cannot answer it

The tempting biological question is:

> "Do the same cells express both CD8 and TIM-3?"

This is called **co-expression**. It means one individual cell has both markers.

That is exactly the kind of claim people want, because CD8 marks cytotoxic T
cells and TIM-3 can be associated with exhaustion. A CD8+TIM-3+ cell sounds like
an exhausted killer T cell.

But with ordinary serial IHC, that claim is not defensible.

### 6.1 The serial-section problem

In serial sections, CD8 and TIM-3 are stained on different physical tissue slices.

Think of a loaf of bread:

- Slice 1 is stained for CD8.
- Slice 2, next to it, is stained for TIM-3.

Even if the slices are adjacent, they are not the same physical plane. A cell in
slice 1 may be absent, cut differently, or replaced by a neighbor in slice 2.

That is called the **z-gap**. The x-y position may look similar, but the z-plane
is different.

### 6.2 Registration does not solve cell identity

**Registration** can align tissue structures between two images. It can say:

> "This vessel, lumen, boundary, or tissue region lines up with that one."

It cannot prove:

> "This segmented object in slice 1 is the same biological cell as that segmented
> object in slice 2."

That is the central scientific limitation.

### 6.3 TIM-3 is not CD8-specific

TIM-3 can appear on several immune cell types, not only CD8 T cells. A TIM-3
positive object near a CD8 object may be:

- an exhausted CD8 T cell,
- a CD4 T cell,
- a regulatory T cell,
- an NK cell,
- a myeloid/dendritic cell,
- or another nearby immune population.

Proximity alone does not identify the cell type.

### 6.4 Compartment mismatch

Different markers may be measured in different parts of the cell:

- CD8 may be measured using a nuclear-adjacent proxy or configured nuclear
  measurement in the current demonstration.
- TIM-3 is membrane-associated.

Comparing "nuclear center" in one section with "membrane ring" in another section
adds another reason not to claim same-cell identity.

### 6.5 The honest conclusion

From serial sections, the honest conclusion is:

> We can study population-level spatial association.

The dishonest overclaim would be:

> These exact cells co-express CD8 and TIM-3.

The project intentionally avoids that overclaim.

---

## 7. Slides 9-10: MNN, why it sounded useful, and why it failed

MNN means **mutual nearest neighbor**.

The idea was simple:

1. For each cell in section A, find the nearest cell in section B after
   registration.
2. For each cell in section B, find the nearest cell in section A.
3. Keep only pairs where both cells choose each other.

That is stricter than one-way nearest-neighbor matching. It removes many
ambiguous matches.

But it still fails for serial-section co-expression.

### 7.1 What MNN actually tells us

MNN tells us:

> "These two detected objects are each other's nearest aligned neighbors."

That is a geometric statement.

### 7.2 What MNN does not tell us

MNN does not tell us:

> "These are the same biological cell."

Why not?

- The sections are different physical slices.
- A cell can be absent or truncated in the next slice.
- A neighboring cell can become the nearest match.
- Dense immune regions create many possible near matches.
- Small registration errors can be large relative to nuclear diameter.

So MNN can create reasonable-looking pairs, but those pairs are not proof of
same-cell identity.

That is why the project moved away from MNN and toward spatial association.

---

## 8. Slide 12: Spatial association as the serial-section alternative

Spatial association asks:

> "Are marker-A positive cells enriched near marker-B positive cells within the
> tissue architecture?"

This is a population question, not a same-cell question.

An everyday analogy:

> Suppose coffee shops and bookstores are often in the same neighborhoods. That
> does not mean a coffee shop is a bookstore. It means the two types of places
> tend to occupy nearby areas.

For serial IHC:

- CD8-positive cells are one population.
- TIM-3-positive cells are another population.
- We ask whether the populations are closer than expected under a suitable chance
  model.

This fits serial sections because it does not require proving that one cell exists
in both sections.

---

## 9. Slides 13-19: Spatial association workflow

The PPT shows this workflow:

1. Pair the sections.
2. Segment both images.
3. Place landmarks.
4. Fit a transform.
5. Certify the registration.
6. Build the shared tissue window.
7. Compute the cross-type K / L(r)-r curve.
8. Run a global curve test.
9. Interpret with the calibrated primary null and CSR diagnostic.
10. Export images, JSON, and provenance.

Let's unpack each idea.

### 9.1 Pair sections

The software must know which two images belong together:

- reference image, usually marker A,
- moving image, usually marker B.

The reference image defines the coordinate system. The moving image is transformed
into that coordinate system.

### 9.2 Segment both images

Each image becomes a point pattern:

- marker-A positive centroids,
- marker-B positive centroids.

A **centroid** is the center point of a detected object.

### 9.3 Place landmarks

A landmark is a corresponding tissue structure visible in both sections, such as:

- a vessel,
- a lumen,
- a boundary,
- a recognizable tissue shape.

The key rule is: landmarks should be structural, not marker-positive cells.
Marker-positive cells are exactly what we want to test, so they should not guide
registration.

### 9.4 Fit a transform

The project uses a **similarity transform**:

- rotation,
- translation,
- one uniform scale factor.

This is intentionally conservative. It preserves the meaning of distance.

The project avoids non-rigid warping for this distance-based statistic because a
bendy warp can force landmarks to match while stretching or shrinking local
distances. That would damage the very distances that the spatial statistic is
trying to measure.

### 9.5 Certify registration

Registration must be measured, not assumed.

The project uses three main certification measurements:

| Measurement | Plain meaning |
|---|---|
| Held-out TRE | If one landmark is left out, how far does the fitted transform miss it? |
| Fit residual | When all landmarks are used, how much mismatch remains? |
| Coverage | How much of the tissue field is supported by reliable landmarks? |

### 9.6 Held-out TRE

TRE means **target registration error**.

The workflow is:

1. Leave one landmark pair out.
2. Fit the transform using the other landmarks.
3. Predict where the left-out landmark should land.
4. Measure the remaining error in µm.
5. Repeat for every landmark.

Plain meaning:

> "If we did not let this point help with the alignment, could the alignment still
> predict it?"

Lower TRE means stronger evidence that the registration is accurate.

### 9.7 Fit residual

Fit residual measures how well the chosen transform explains all landmark pairs.

Plain meaning:

> "After the best rotation, shift, and scale, how much mismatch is still left?"

A low residual means one similarity transform can explain the landmark geometry.
A high residual means the tissue may be deformed or the landmarks may not
correspond consistently.

### 9.8 Coverage

Coverage asks whether the reliable landmarks support enough of the image.

Six excellent landmarks in one tiny corner do not certify the entire field. They
may certify a local region, but not the whole image.

### 9.9 Certification thresholds

The PPT's working criterion is:

- at least 6 landmarks,
- median held-out TRE ≤ 5 µm,
- fit residual ≤ 5 µm,
- for local certification, a coherent passing subset whose hull covers at least
  10% of the field.

### 9.10 Certification verdicts

| Verdict | Meaning | What happens |
|---|---|---|
| CERTIFIED | Global landmark set passes. | The supported field can be analyzed. |
| LOCALLY_CERTIFIED | A coherent region passes, not the whole field. | Analysis is restricted to that ROI. |
| DEFORMED | Landmarks correspond, but one similarity transform cannot fit them. | Do not warp; stop the spatial test. |
| NOT_CERTIFIABLE | Too few landmarks or errors too large. | No distance-based association result is produced. |

This is a **fail-closed** design. If the software cannot prove alignment quality,
it refuses to produce a confident biological result.

### 9.11 Why automatic registration QC was not enough

The project tested several automatic approaches:

- ORB/SIFT nuclear feature matching,
- patch phase correlation,
- dense-edge Chamfer,
- tissue-outline Chamfer,
- auto-proposed lumens,
- tissue-overlap scoring.

The important finding was that these methods could look plausible while being
insensitive to the 10-50 µm scale required by the statistic.

Example from the PPT:

- a deliberate 30 µm shift could be reported as about 0.2 µm by a texture-based
  method.

That is why the current certification relies on trusted structural landmarks and
explicit error measurement.

### 9.12 Build the shared tissue window

The statistic should only use tissue that exists in both physical sections.

The project builds:

```text
analysis window = tissue mask A ∩ transformed tissue mask B
```

This is the intersection of:

- the reference tissue mask,
- the moving tissue mask after registration.

Why this matters:

> If tissue exists in the CD8 section but is missing from the TIM-3 section, then
> no TIM-3 cells can appear there. Treating that as true biological absence would
> create a fake segregation result.

Internal holes and lumens are preserved as non-tissue because empty space is not
somewhere a cell can be.

---

## 10. Slides 20-21: Cross-type Ripley K and L(r)-r

After registration and masking, we have:

- a set of A-positive cell centers,
- a set of B-positive cell centers,
- a certified tissue window.

Now we need to measure whether A and B are unusually near each other.

### 10.1 Cross-type Ripley K formula

The PPT gives:

```text
K_AB(r) = |W| / (N_A N_B) × Σ 1[d(a_i, b_j) ≤ r]
```

Meaning of each piece:

| Symbol | Meaning |
|---|---|
| A and B | The two positive-cell populations. |
| a_i | One cell from population A. |
| b_j | One cell from population B. |
| d(a_i, b_j) | Distance between those two cells. |
| r | Search radius, such as 10, 20, or 50 µm. |
| 1[d ≤ r] | Count 1 if the pair is within radius r, otherwise 0. |
| Σ | Add up the count across all A-B pairs. |
| N_A, N_B | Number of A and B cells. |
| W | The analysis tissue window. |
| |W| | Physical area of that window. |

Plain meaning:

> "For every A cell, count how many B cells fall within radius r. Normalize that
> count so fields with different tissue areas and cell counts can be compared."

The project uses `cKDTree`, a fast nearest-neighbor search structure, so this pair
counting can be done efficiently.

### 10.2 Why K is cumulative

K is cumulative because the count includes all pairs closer than or equal to r.

At r = 10 µm, it counts close neighbors.

At r = 50 µm, it counts everything within a larger neighborhood, including the
pairs already counted at smaller radii.

That is why the output is a curve, not a single number.

### 10.3 L(r)

The L function is:

```text
L(r) = sqrt(K(r) / π)
```

The purpose of L is readability. K grows roughly like area, so it can be harder
to interpret directly. L converts K back into distance-like units.

### 10.4 L(r)-r

The plotted curve is:

```text
L(r) - r
```

Interpretation:

| Curve position | Plain meaning |
|---|---|
| L(r)-r = 0 | Compatible with independence. |
| L(r)-r > 0 | More cross-neighbors than expected; association. |
| L(r)-r < 0 | Fewer cross-neighbors than expected; segregation. |

### 10.5 Why the analysis band is 10-50 µm

The global test focuses on 10-50 µm.

Below about 10 µm, cell centers physically cannot overlap. This is called
**hard-core exclusion**. It can force negative values even when there is no
biological avoidance.

Above about 50 µm, the curve increasingly reflects broad tissue compartments
rather than local cell-scale association.

So 10-50 µm is the biologically meaningful band for the current analysis.

---

## 11. Slide 22: DCLF, the one p-value for the whole curve

A curve creates a statistical trap. If we test every radius separately and then
pick the most exciting radius, we inflate false positives.

The project uses the **DCLF** global envelope test:

```text
u = Σ_r [L(r) - L_mean(r)]²
```

Plain meaning:

> "At each radius, measure how far the curve is from the average null curve.
> Square those distances so positive and negative deviations both count. Add the
> squared distances across the 10-50 µm band."

The result is one number, `u`, for the entire curve.

Then the p-value is:

```text
p = (1 + number of null u values ≥ observed u) / (1 + number of permutations)
```

Plain meaning:

> "If there were no real association beyond the null expectation, how often would
> a simulated curve deviate at least as much as the real curve?"

Why the `1 +` terms? They prevent a p-value of exactly zero when using a finite
number of simulations. With 999 simulations, the smallest possible p-value is:

```text
1 / (1 + 999) = 0.001
```

The PPT reports that under a synthetic true null, the DCLF false-positive rate was
**0.045** and the mean p-value was **0.515**. That is what we want: a fair test
should produce p < 0.05 about 5% of the time under a true null, and p-values
should average around 0.5.

---

## 12. Slides 23-29: The null model problem

The phrase "more than expected by chance" hides the hardest part:

> What does "chance" mean in real tissue?

### 12.1 The wrong null can create fake biology

Suppose both cell types independently prefer the same tissue region, such as an
inflamed margin.

They will appear near each other, but not because one attracts the other. They are
near each other because both like the same neighborhood.

This is **shared tissue preference**.

An honest null model must preserve that shared architecture. Otherwise it will
mistake shared preference for direct association.

### 12.2 Homogeneous CSR

CSR means **complete spatial randomness**.

In homogeneous CSR:

- A is held fixed.
- B is redrawn uniformly across the tissue window.

This is useful as a simple diagnostic baseline, but it is too weak as the primary
test. It removes B's real tissue preference. If real B naturally prefers the same
regions as A, the observed pattern will look artificially associated.

The PPT reports a false-positive rate of **1.00** under shared preference for the
homogeneous CSR baseline.

Plain meaning:

> In the shared-preference stress test, CSR called association essentially every
> time, even though the constructed truth had no direct A-B interaction.

### 12.3 Retired smoothed inhomogeneous resampling

The project tried a smoothed intensity-resampling null:

1. Estimate where B tends to live.
2. Draw new B points from that estimated surface.
3. Compare the observed curve with the simulated curves.

It sounded right, but it failed calibration.

Why it failed:

- The observed B pattern estimated its own intensity surface.
- The simulated patterns were smoother than the observed pattern.
- The bandwidth overlapped the same 10-50 µm band being tested.
- The estimator was not the correct intensity-reweighted cross-K.

The PPT reports a false-positive rate of **0.87** under shared preference. That is
far too high.

### 12.4 Retired toroidal shift

The toroidal-shift null slides the entire B pattern by a random offset and wraps
points around the rectangular boundary.

It preserves B's internal clustering, but it assumes the tissue is stationary:
that the pattern could plausibly appear anywhere in the window.

Real tissue violates that assumption. Vessels, margins, lumens, and compartments
are not interchangeable across the image.

The PPT reports a false-positive rate of **0.85** under shared preference. That is
also far too high.

### 12.5 Why random labeling is invalid for serial sections

Random labeling would pool all A and B points and shuffle the labels.

That can be valid for a single section where every marker was measured on the
same physical cells. It is not valid for serial sections.

In serial sections:

- A points came from one physical slide.
- B points came from another physical slide.
- The marker label is tied to the section where the point was observed.

Shuffling labels would invent measurements that never happened. It would assign a
marker to coordinates where that marker was never observed.

So the production null repositions one population while preserving each marker's
own data-generating process.

### 12.6 Production primary: intensity-reweighted inhomogeneous cross-K

The production primary statistic is the calibrated **intensity-reweighted
inhomogeneous cross-K**.

The PPT gives:

```text
K_inhom_AB(r) = 1/|W| × Σ 1[d(a_i,b_j) ≤ r] / [λ_A(a_i) λ_B(b_j)]
```

Meaning of the new terms:

| Term | Meaning |
|---|---|
| λ_A(a_i) | Estimated expected density of A cells at the location of A cell i. |
| λ_B(b_j) | Estimated expected density of B cells at the location of B cell j. |
| 1 / [λ_A λ_B] | A correction weight that down-weights pairs in places both populations already like. |

Plain meaning:

> "If two populations are close only because both love the same tissue region, we
> should not count that closeness as strong evidence. We discount pairs in already
> dense regions and ask whether proximity remains after each population's own
> tissue preference is accounted for."

Under independence after accounting for architecture, the expected curve behaves
like the theoretical null. That is why this is the primary test.

### 12.7 Leave-one-out intensity

When estimating λ, the software uses **leave-one-out** intensity.

Plain meaning:

> "Do not let a cell make its own exact location look artificially popular."

If a cell contributes to the intensity estimate at its own point, every cell gives
itself a small home-field advantage. Leave-one-out removes that self-attraction
bias.

### 12.8 Bootstrap symmetry

For each simulated B pattern:

1. Hold A fixed.
2. Draw a new B* from the estimated B intensity surface.
3. Re-estimate intensity for B*.
4. Recompute the reweighted K curve.
5. Recompute DCLF.

This is important because observed and simulated patterns must be processed
symmetrically. Otherwise the observed pattern could be rougher or more clustered
than the null simply because of how the test was built.

### 12.9 Bandwidth

The production configuration uses a **75 µm Gaussian bandwidth** with
leave-one-out intensity.

Plain meaning:

> The intensity surface is meant to capture broad tissue architecture, not the
> tiny cell-to-cell interaction scale.

This assumes that the tissue architecture being adjusted for is coarser than the
10-50 µm interaction band. If architecture exists at the same scale as individual
cells, no statistic can perfectly separate "shared tissue preference" from "real
cell-cell attraction" without extra tissue covariates.

---

## 13. Slides 30-33: Calibration and demonstration

### 13.1 Calibration selected the production configuration

The PPT reports a 500-run calibration:

| Regime | Result |
|---|---:|
| Shared-preference size | 3.2% |
| Uniform CSR size | 6.4% |
| Power for 7 µm attraction | 100% |
| Power for 25 µm attraction | 99.2% |

Plain meaning:

> Under a true no-interaction scenario, the method calls significance about as
> often as it should. When a real attraction is planted, it detects it.

This is what a good statistical test needs:

- low false-positive rate under the null,
- high power when a real effect exists.

### 13.2 Cross-validation against R spatstat

The custom Python estimator was checked against **spatstat**, a reference spatial
statistics package in R.

The PPT reports:

- max L relative difference around **1.4 × 10^-10** on a real CODEX spot,
- synthetic cases matched at about **10^-14**.

Plain meaning:

> The Python implementation is not doing a different calculation by accident. It
> matches the reference estimator to numerical precision when given identical
> inputs.

### 13.3 Why no edge correction in the production test

Raw K values can be affected by image boundaries because cells near the edge have
part of their possible neighborhood outside the window.

The project uses the same uncorrected estimator for:

- the observed curve,
- every null-simulated curve.

Because both sides are treated the same way, the boundary undercount cancels in
the DCLF rank comparison. Validation showed translation edge correction did not
change calibration or power.

Plain meaning:

> Edge correction changed raw K, but it did not improve the actual p-value test.
> So the simpler symmetric estimator was kept.

### 13.4 Lung-lesion demonstration

The PPT includes a locally certified demonstration:

- CIMA/ANHIR lung-lesion_1,
- Ki67: 1,169 cells,
- proSPC: 3,342 cells,
- locally certified ROI,
- 10 of 78 landmarks support the ROI,
- CSR p = 0.001,
- reweighted primary p ≈ 0.325.

Interpretation:

> Under the weak uniform-scattering baseline, the pattern looks associated. After
> correcting for each population's tissue intensity, the global association is not
> significant over 10-50 µm.

That is exactly why the calibrated primary matters. It can demote a flashy CSR
association into a shared-preference explanation.

---

## 14. Slides 34-37: When true co-expression is possible

True same-cell co-expression becomes possible when multiple markers are measured
on the **same physical tissue section**.

There are two main routes:

1. **Multiplex IHC / multiplex imaging**: multiple markers are stained and imaged
   on one section.
2. **Same-section sequential IHC / restaining**: stain, image, strip or reset,
   then stain the same section again.

The key difference from serial sections:

> The same physical cells can remain in the coordinate frame.

That allows a different endpoint:

- A-only,
- B-only,
- double-positive,
- double-negative.

But same-section provenance alone is not enough. The project still checks:

- image dimensions,
- manual shared-coordinate certification,
- structural diagnostics,
- segmentation quality,
- marker threshold validity.

Equal dimensions do not prove that every local cell coordinate still corresponds.
Stripping, deformation, cropping, and mismatched public files can still break
correspondence.

---

## 15. Slides 35-39: Restained same-cell workflow

The restained workflow in the PPT is:

1. Load three captures:
   - hematoxylin reference,
   - marker A AEC image,
   - marker B AEC image.
2. Check geometry:
   - same width,
   - same height,
   - operator confirmation of shared coordinates.
3. Optionally preprocess faint hematoxylin.
4. Segment once on the hematoxylin reference.
5. Measure marker A in the chosen compartment.
6. Measure marker B in the same segmented polygons or chosen compartment.
7. Apply operator-supplied thresholds.
8. Export per-cell table, overlay, JSON, and statistics.

### 15.1 Why segment once?

If the same physical section is used, the strongest approach is to segment the
cells once and reuse those same cell objects for both markers.

Plain meaning:

> "Draw the cell map once, then ask what marker A and marker B measure inside each
> same cell object."

This avoids the serial-section problem because we are no longer trying to match
different cells across different tissue planes.

### 15.2 AEC instead of DAB

The restaining demonstration uses **AEC**, another chromogenic stain. The logic is
similar to DAB:

- stronger marker signal creates stronger color,
- optical density is measured,
- a threshold turns continuous signal into positive/negative.

But AEC thresholds are not copied from DAB thresholds. They must be supplied for
the AEC images.

### 15.3 Faint-nucleus preprocessing

The PPT describes optional hematoxylin preprocessing:

1. Deconvolve the image to isolate hematoxylin optical density.
2. Stretch the positive hematoxylin signal between fixed percentiles.
3. Reconstruct an H-only RGB image.
4. Segment once.
5. Validate against expert nuclear masks when available.

Plain meaning:

> "Make faint nuclei easier for the segmentation model to see, but save the
> intermediate image and validate whether it actually helped."

The audit found a trade-off:

- preprocessing recovered additional true nuclei,
- but also added false detections.

So it is not automatically "better." It is an explicit option with measurable
consequences.

---

## 16. Slides 40-41: The 2×2 same-cell table and its statistics

For same-section restaining, each cell can be classified into four boxes:

| | Marker B positive | Marker B negative |
|---|---:|---:|
| Marker A positive | double-positive | A-only |
| Marker A negative | B-only | neither |

The PPT's Case1_M1_0_0 nuclear demonstration table is:

| | FOXP3+ | FOXP3- |
|---|---:|---:|
| CD8+ | 11 | 40 |
| CD8- | 0 | 388 |

Totals:

- 439 segmented cells,
- 51 CD8-positive,
- 11 FOXP3-positive,
- 11 double-positive.

The PPT reports:

- 8.61× over independence expectation,
- Fisher p = 1.85e-11,
- φ = 0.442.

### 16.1 Expected double positives under independence

If CD8 and FOXP3 were independent, the expected double-positive count is:

```text
expected double positives = (CD8+ total × FOXP3+ total) / total cells
```

For the PPT table:

```text
expected = 51 × 11 / 439 ≈ 1.28
observed = 11
enrichment = observed / expected ≈ 8.61
```

Plain meaning:

> "If CD8 and FOXP3 positivity were unrelated in this table, we would expect about
> 1.28 double-positive cells. We observed 11."

### 16.2 Fisher exact test

Fisher exact test asks:

> "If the two markers were independent, how surprising is this 2×2 table?"

It is especially useful when some counts are small, such as the `0` in the
FOXP3+/CD8- box.

The p-value has the usual meaning:

> "Under independence, how often would chance produce a table this extreme or
> more extreme?"

A very small p-value means the table is unlikely under independence.

### 16.3 Phi coefficient

The phi coefficient, written **φ**, is a signed strength measure for two binary
variables.

For a table:

```text
          B+   B-
A+        a    b
A-        c    d
```

the formula is:

```text
φ = (a d - b c) / sqrt((a+b)(c+d)(a+c)(b+d))
```

Interpretation:

| φ value | Meaning |
|---:|---|
| +1 | Perfect positive association. |
| 0 | Little or no association. |
| -1 | Perfect negative association. |

For the PPT table, φ = 0.442, a moderate positive association.

### 16.4 BH-FDR correction

BH-FDR means **Benjamini-Hochberg false discovery rate** correction.

It is used when many tile-level p-values are tested.

Plain meaning:

> "If we test many tiles, some may look significant by luck. BH-FDR adjusts the
> p-values so the reported significant set is less likely to be a pile of random
> false alarms."

The corrected p-value is often called a **q-value**.

### 16.5 Important scope note

The PPT labels the Case1_M1_0_0 result as a **current nuclear demonstration** and
also states its scope:

> The software computes a strong table-level association at the supplied
> thresholds; the dataset does not include expert CD8/FOXP3-positive cell labels
> for threshold validation.

`ihc.md` adds an additional caution: compartment choice is load-bearing. A marker
measured in the wrong compartment can create a misleading result. Therefore this
section should be presented as a software/statistics demonstration at supplied
thresholds, not as a final biological claim.

---

## 17. Slides 42-44: Restaining validation and what it means

### 17.1 Three-tile segmentation pilot

The PPT reports a three-tile pilot from HNSCC Case 1:

- Case1_M1_0_0,
- Case1_M1_0_1,
- Case1_M1_1_0,
- 512×512 tiles,
- 0.5 µm/pixel,
- expert-corrected nuclear masks.

What it tests:

> Object-level nuclear detection against released segmentation masks.

What it does not test:

> Expert CD8 or FOXP3 positive/negative marker labels.

The pilot finding:

- preprocessing recovered 37 additional true nuclei,
- preprocessing added 69 false detections,
- recall increased by 3.2 percentage points,
- precision decreased by 5.2 percentage points,
- F1 changed from 0.854 to 0.847.

Plain meaning:

> Preprocessing helped find more real nuclei but also made more mistakes. The
> combined score did not clearly improve. It is a documented trade-off.

### 17.2 Full external audit

The PPT reports the full audit across 268 tiles and eight patients:

| Endpoint | Result |
|---|---:|
| Reference nuclei | 95,519 |
| Predicted nuclei | 85,336 |
| Micro-F1 | 0.776 |
| Precision | 0.822 |
| Recall | 0.734 |
| Median tile F1 | 0.786 |
| Spearman ρ between structure correlation and F1 | 0.740 |

### 17.3 Spearman rho

Spearman rho, written **ρ**, measures whether two quantities move together
monotonically.

Plain meaning:

> "When structural correspondence is better, does segmentation F1 tend to be
> better too?"

Here ρ = 0.740 means a strong positive relationship: tiles with better structural
correspondence tended to have better measured F1.

### 17.4 Pixel AUC for marker localization

The PPT reports mIF foreground had higher AEC OD in most tiles:

- CD8: 261/264 tiles, median pixel AUC 0.939.
- FOXP3: 266/268 tiles, median pixel AUC 0.777.

**AUC** means area under the ROC curve.

Plain meaning:

> "If I randomly pick one foreground pixel and one background pixel, how often
> does the marker measurement rank the foreground pixel higher?"

Interpretation:

- AUC = 0.5 means no better than random.
- AUC = 1.0 means perfect separation.

So CD8 localization looked strong overall; FOXP3 localization was positive but
more variable.

### 17.5 Binary marker sensitivity results

Against a non-expert mIF intensity proxy, the PPT reports:

- CD8 F1 = 0.512,
- FOXP3 F1 = 0.543.

Plain meaning:

> The AEC signal often localizes to the right broad regions, but the fixed binary
> positive/negative thresholds are not fully validated as cell-level marker
> truth.

This is why the project says the audit validates:

- pipeline execution,
- nuclear segmentation behavior,
- broad AEC localization,

but not:

- expert marker-positive thresholds,
- final CD8/FOXP3 biology,
- CD8/TIM-3 biology.

### 17.6 What worked, what failed, and the current restaining sequence

What worked:

- one segmentation can be reused across same-section marker captures,
- per-cell measurements can be exported,
- CSV / GeoJSON / overlay / JSON artifacts are produced,
- 2×2 statistics are computed.

Mixed result:

- faint-H preprocessing increases recall but can reduce precision.

Failed assumption:

- equal dimensions and same-section provenance do not guarantee local
  correspondence in every public tile.

Remaining input gap:

- public masks identify nuclei,
- they do not provide expert CD8/FOXP3 positive cell classes.

Current operating sequence:

1. Confirm same field.
2. Segment hematoxylin once.
3. Measure marker signals in the correct compartments.
4. Apply entered thresholds.
5. Compute per-tile Fisher/phi.
6. Apply BH-FDR across tiles.
7. Inspect overlays and provenance.

---

## 18. Slides 45-47: Validation layers

The PPT ends by separating validation layers. This is essential because one
successful validation does not validate every part of the project.

### 18.1 Statistical validation

| Test | Dataset | Meaning |
|---|---|---|
| Cross-K implementation | Synthetic coordinates | Brute-force pair counting exactly matches cKDTree. |
| DCLF calibration | Synthetic null / clustered / separated patterns | p-values behave correctly under known truth. |
| Old-null stress test | 500 shared-preference realizations | Old nulls were unsuitable for shared architecture. |
| Reweighted calibration | Shared preference, uniform, attraction | Production 75 µm leave-one-out design is calibrated. |
| R cross-validation | Synthetic + real CODEX spot | Python estimator matches spatstat. |
| Real-data controls | Schürch CRC CODEX | Shows how shared-architecture adjustment changes real conclusions. |

Plain meaning:

> The math engine was tested separately from the biological datasets. The
> statistics behave correctly when the truth is known.

### 18.2 Registration validation

| Test | Dataset | Meaning |
|---|---|---|
| Known-transform unit test | Synthetic shifted structural image | The transform engine works when correspondence exists. |
| Target cohort feasibility | 052526 CD8/TIM-3 serial-section pairs | No pair was certified for 10-50 µm association. |
| ANHIR/CIMA public landmarks | 83 stain pairs | The gate restricts analysis when non-rigid deformation dominates. |
| Independent annotator check | Lung-lesion_3 HE vs proSPC | Rejection reproduced with independent landmark evidence. |
| Local positive example | Lung-lesion_1 Cc10 vs proSPC | A supported local ROI can pass. |
| HyReCo status | Consecutive and restained sections | Relevant future dataset, but gated/large archive blocked real run. |

Plain meaning:

> The registration system can prove when it should not trust a pair. That is just
> as important as proving when it can trust a pair.

### 18.3 Dataset map

The PPT's dataset map separates purposes:

| Dataset | Used for |
|---|---|
| HNSCC mIF/mIHC comparison v2 | Restained same-section nuclear segmentation and AEC localization audit. |
| 052526 CD8/TIM-3 | Target serial-section feasibility; no certified pair. |
| ANHIR/CIMA | Public expert-landmark registration certification. |
| Schürch CRC CODEX | Real point-pattern statistics validation, not serial registration. |
| Synthetic null regimes | Calibration, power, known truth. |
| R spatstat | Reference estimator equivalence. |
| Lung-lesion Ki67/proSPC | Locally certified spatial demonstration. |
| Case1_M1_0_0 restained folder | Same-cell software demonstration. |

The main lesson:

> Each dataset validates a different layer. None of them magically validates every
> biological claim.

---

## 19. Formula cheat sheet

This section repeats the main formulas in one place.

### 19.1 Pixel size

```text
distance_um = distance_px × pixel_size_um_per_px
```

Meaning:

> Converts image pixels into real microscope distance.

### 19.2 Precision

```text
precision = TP / (TP + FP)
```

Meaning:

> Of the objects the computer called positive/found, how many were correct?

### 19.3 Recall

```text
recall = TP / (TP + FN)
```

Meaning:

> Of the expert objects, how many did the computer find?

### 19.4 F1

```text
F1 = 2 × precision × recall / (precision + recall)
```

Meaning:

> Balanced score that gets worse if either precision or recall is poor.

### 19.5 Dice

```text
Dice = 2 × overlap / (predicted area + expert area)
```

Meaning:

> How strongly two masks overlap.

### 19.6 IoU

```text
IoU = overlap / union
```

Meaning:

> Of everything either mask marked, how much did both masks agree on?

### 19.7 Cross-type Ripley K

```text
K_AB(r) = |W| / (N_A N_B) × Σ 1[d(a_i,b_j) ≤ r]
```

Meaning:

> Count A-B neighbor pairs within radius r and normalize by tissue area and cell
> counts.

### 19.8 L function

```text
L(r) = sqrt(K(r) / π)
```

Meaning:

> Converts K into an easier distance-like scale.

### 19.9 L(r)-r

```text
L(r) - r
```

Meaning:

> Positive suggests association; negative suggests segregation; zero suggests
> independence.

### 19.10 DCLF curve statistic

```text
u = Σ_r [L(r) - L_mean(r)]²
```

Meaning:

> One total distance between the observed curve and the null mean curve across
> the tested radius band.

### 19.11 DCLF p-value

```text
p = (1 + # null u ≥ observed u) / (1 + permutations)
```

Meaning:

> How often null simulations are at least as extreme as the observed curve.

### 19.12 Intensity-reweighted cross-K

```text
K_inhom_AB(r) = 1/|W| × Σ 1[d(a_i,b_j) ≤ r] / [λ_A(a_i) λ_B(b_j)]
```

Meaning:

> Discounts pairs in areas where both populations are already expected to be
> dense because of tissue architecture.

### 19.13 Expected double positives

```text
expected double positives = (A+ total × B+ total) / total cells
```

Meaning:

> How many double-positive cells we would expect if the two markers were
> independent.

### 19.14 Enrichment ratio

```text
enrichment = observed double positives / expected double positives
```

Meaning:

> How many times more double-positive cells were observed than independence would
> predict.

### 19.15 Phi coefficient

```text
φ = (a d - b c) / sqrt((a+b)(c+d)(a+c)(b+d))
```

for:

```text
          B+   B-
A+        a    b
A-        c    d
```

Meaning:

> Signed strength of association between two yes/no labels.

### 19.16 Fisher exact p-value

No simple one-line arithmetic formula is usually written by hand, but its meaning
is:

> Under independence, how likely is this 2×2 table or a more extreme one?

It is preferred when cell counts are small.

### 19.17 Benjamini-Hochberg FDR

Meaning:

> When many p-values are tested, adjust them so the expected fraction of false
> discoveries among the called-significant results is controlled.

The adjusted value is often called **q**.

### 19.18 Spearman rho

Meaning:

> Measures whether two quantities tend to rise and fall together, using ranks
> rather than raw values.

`ρ = 1` means perfect same-direction ranking, `ρ = 0` means no monotonic pattern,
and `ρ = -1` means perfect opposite ranking.

### 19.19 AUC

Meaning:

> If one positive/foreground example and one negative/background example are
> chosen at random, AUC is the probability the score ranks the positive one
> higher.

AUC = 0.5 is random. AUC = 1.0 is perfect.

---

## 20. The honest bottom line

The project is not just trying to get a nice-looking answer. It is trying to
avoid false confidence.

The strongest defensible statements are:

1. **Quantification** can count segmented cells and classify them by supplied
   stain thresholds, while exporting auditable overlays and tables.
2. **Segmentation validation** checks whether nuclei are found; it does not prove
   marker thresholds.
3. **Serial-section CD8/TIM-3 co-expression cannot be claimed** because the two
   stains are on different physical tissue planes.
4. **Spatial association is the correct serial-section alternative** because it
   asks whether populations occupy nearby neighborhoods, not whether exact cells
   are double-positive.
5. **The production spatial statistic corrects for shared tissue preference**
   using intensity-reweighted inhomogeneous cross-K plus a DCLF global curve test.
6. **Registration must be certified**; if alignment cannot be proven at the needed
   micrometre scale, the result is blocked or restricted to a certified ROI.
7. **True same-cell co-expression requires same-section data**, such as multiplex
   IHC or same-section restaining, and even then thresholds and coordinate
   correspondence need validation.
8. **The validation story is layered**: statistics, registration, segmentation,
   marker localization, and biological marker thresholds are separate things.

The simple way to explain the whole project is:

> We built an analysis pipeline that can count stained cells, compare spatial
> neighborhoods carefully, and show when the data is not strong enough for a
> stronger claim. The most important result is not just the number it produces,
> but the boundary it draws around what that number is allowed to mean.
