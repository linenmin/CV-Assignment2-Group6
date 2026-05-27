# DM-0: Chapter 0 — Title, Abstract, TL;DR

**Chapter goal**: open the report with a 30-second snapshot of what we
did, who we are, and what the headline numbers are, so a TA reading
linearly knows immediately what to expect from the rest of the
notebook.

## Proposed cell list

| # | Type | One-line content |
|---|---|---|
| 0.1 | markdown | Title block: course code, assignment name, group number, member list |
| 0.2 | markdown | Abstract (~150 words): one paragraph covering data, approach, headline results, and one sentence on the deployment-readiness finding |
| 0.3 | markdown | TL;DR table: headline numbers for each of the three tasks + the Kaggle leaderboard score |
| 0.4 | markdown | Reading guide: 4-line section map + note that all heavy code lives in the bundled Kaggle dataset |

## Figures / tables to embed

- **No figures** in Chapter 0. Visual content starts in Chapter 1
  (data overview).
- **One markdown TL;DR table** (cell 0.3). Data source: composed from
  numbers already verified in `plan/findings.md`.

## TL;DR table content (draft)

| Task | Headline metric | Value | Source |
|---|---|---|---|
| 2.1 Image classification | Kaggle classification Dice | **0.898** | ConvNeXt-Small 320, threshold-calibrated, +TTA |
| 2.2 Semantic segmentation | Kaggle public score | **0.89139** | EoMT-DINOv3 V17 + multi-scale TTA, merged with classification |
| 2.3 Adversarial attack on classifier | PGD targeted success rate | **100%** | aeroplane target, eps=0.03 normalised |
| 2.3 Adversarial attack on segmenter | CosPGD prediction change rate | **99.6%** | untargeted, eps=6/255, 20 steps |
| 2.4 Real-world reality check | Cityscapes 5-class transferable mIoU | **0.509** | overlap classes, no Cityscapes training |

## Text tone

- Plain English, present tense ("we train", "we evaluate"). No bullet
  spam in the abstract — it should read like an academic abstract.
- Target word count: abstract ~150 words, reading guide ~50 words.
- No filler ("In this work, we present..."). Open with a concrete
  statement of what the system does.

## Demo code

- **None** for Chapter 0. Code starts in Chapter 1.

## Sub-options requiring user input

### Sub-option A: Author display
- **A1 (recommended)**: Show full member list with both English names
  and roles (e.g. "Enmin Lin — segmentation lead"). Roles help the TA
  attribute work.
- A2: Just list names without roles.
- A3: Pull author block straight from the Image-Classification README
  format (names only, no roles).

### Sub-option B: TL;DR table emphasis
- **B1 (recommended)**: Five rows as drafted above — separate row for
  each attack track so the strongest numbers don't hide behind one
  collapsed "attacks" row.
- B2: Four rows — collapse the two attack rows into one.

### Sub-option C: Reading guide tone
- **C1 (recommended)**: Brief navigation sentence + section list +
  one-line note about the bundled Kaggle dataset.
- C2: Skip the reading guide; let the section headers do the work.

## Estimated effort
- ~10 minutes to draft + render + self-review. Zero figure generation.

## What I need from the user
- Approve / adjust the chapter scope.
- Pick A / B / C variants (or accept the recommended defaults).
- Confirm or correct the member list and any role attribution.

## What I will deliver after approval
- A `notebook/group6_final.ipynb` with the four cells above filled.
- A screenshot read-back to confirm markdown renders cleanly.
- Updated `STORY_MAP.md` chapter status to "draft v1".
- Discord ping with the verification request.
