#!/usr/bin/env python3
"""
anatomy_prior_probe.py
======================
Does MedGemma read the image, or recite abdominal anatomy from the z-position
prior? This probe breaks the z <-> present-organ correlation by sampling slices
from three strata so that correct answers REQUIRE looking at the pixels.

Strata (per volume, per focus organ)
------------------------------------
  transition        : the most-superior and most-inferior z at which the organ
                      exists (it appears / disappears). Marginal -> hard cases.
  out_of_range      : slices well outside the organ's z-range (ask about liver on
                      a pelvic slice, etc.) -> clean true negatives / fabrication.
  prior_consistent  : mid-range slices where presence is unsurprising -> baseline.

Headline metric (later steps): accuracy drop from prior_consistent -> transition
/ out_of_range, per format. That gap is the language-prior effect.

Query structure
---------------
Per slice we ask about a small BALANCED set (~2 present, ~2 absent, >=1 a
boundary organ), NOT all 13 classes (that invites checklist prior-recitation).
The BTCV class vocabulary is given to the model so it knows the option space.

THIS FILE IS STEP 1 ONLY: label loader, present-organ-per-z, the three-strata
sampler, and the balanced query builder, plus a --dry_run that prints the
sampled (z, stratum, present-organs, queried-set) table and dumps debug PNGs.
NO model calls. Steps 2 (prompts + model wrapper) and 3 (scoring + McNemar +
patient-clustered bootstrap CIs) are added after this is eyeballed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# BTCV / Synapse 13-class vocabulary (verified against data/btcv label0001)
# --------------------------------------------------------------------------- #

BTCV_LABELS = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}
NAME_TO_ID = {v: k for k, v in BTCV_LABELS.items()}
VOCAB = list(BTCV_LABELS.values())

# Organs used as the "focus" of strata. Long midline vessels (aorta, IVC) and the
# tiny adrenals/vein make poor focus organs (no clean transition / too small), but
# they remain in VOCAB as distractors.
DEFAULT_FOCUS_ORGANS = (
    "liver", "spleen", "stomach", "pancreas",
    "gallbladder", "left_kidney", "right_kidney", "esophagus",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    images_dir: str = str(PROJECT_ROOT / "data/btcv/RawData/Training/img")
    labels_dir: str = str(PROJECT_ROOT / "data/btcv/RawData/Training/label")
    out_dir: str = str(PROJECT_ROOT / "results/medgemma_anatomy_prior")

    focus_organs: tuple = DEFAULT_FOCUS_ORGANS
    # Hysteretic thresholds (knob 4): an organ counts as PRESENT only at
    # >= presence_min_pixels; a slice qualifies as a near-miss NEGATIVE only if
    # the focus has < near_miss_hard_zero voxels (a true hard zero). The gap
    # between them shields the negatives from partial-volume boundary flicker.
    presence_min_pixels: int = 50
    near_miss_hard_zero: int = 5
    n_prior_consistent: int = 3     # mid-range baseline slices per organ (knob 1)
    near_miss_min_offset: int = 2   # near-miss negatives: 2..4 slices past a boundary
    near_miss_max_offset: int = 4
    out_of_range_margin: int = 8    # slices beyond the z-range to count as "far"
    n_query_present: int = 2
    n_query_absent: int = 2
    max_cases: int = 2
    seed: int = 0

    # windowing (abdominal soft tissue) + orientation match the other probes
    window_center: float = 40.0
    window_width: float = 400.0

    save_debug_images: bool = True

    # --- step 2: model + run control ---
    model_id: str = str(PROJECT_ROOT / "models/medgemma-27b-it")
    max_new_tokens: int = 320
    do_pan_and_scan: bool = True
    limit_n: int = 0          # >0 -> smoke test on the first N slices only
    dry_run: bool = False     # step-1 strata table only, no model
    n_boot: int = 2000        # patient-clustered bootstrap iterations


# --------------------------------------------------------------------------- #
# Volume loading / rendering (orientation matches scripts/severity_calibration.py)
# --------------------------------------------------------------------------- #

def load_volume(path: str) -> np.ndarray:
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj)


def window_to_uint8(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2.0, center + width / 2.0
    clipped = np.clip(slice_hu.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def to_radiological(arr2d: np.ndarray) -> np.ndarray:
    return np.fliplr(np.rot90(arr2d, k=1))


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


# --------------------------------------------------------------------------- #
# Present-organ-per-z
# --------------------------------------------------------------------------- #

def organ_areas(label_vol: np.ndarray) -> dict[str, np.ndarray]:
    """organ name -> per-z voxel-count array (length = n_slices)."""
    return {name: (label_vol == oid).sum(axis=(0, 1))
            for oid, name in BTCV_LABELS.items()}


def z_ranges(areas: dict[str, np.ndarray], thr: int) -> dict[str, tuple[int, int]]:
    """organ -> (z_min, z_max) where area >= thr; absent organs omitted."""
    out = {}
    for name, a in areas.items():
        present_z = np.where(a >= thr)[0]
        if present_z.size:
            out[name] = (int(present_z.min()), int(present_z.max()))
    return out


def present_at_z(areas: dict[str, np.ndarray], z: int, thr: int) -> set[str]:
    return {name for name, a in areas.items() if a[z] >= thr}


def boundary_organs_at_z(ranges: dict[str, tuple[int, int]], z: int, tol: int = 1) -> set[str]:
    """Organs whose appearance/disappearance slice is within +/-tol of z."""
    out = set()
    for name, (zmin, zmax) in ranges.items():
        if abs(z - zmin) <= tol or abs(z - zmax) <= tol:
            out.add(name)
    return out


# --------------------------------------------------------------------------- #
# Three-strata sampler
# --------------------------------------------------------------------------- #

@dataclass
class SliceSample:
    case: str
    z: int
    stratum: str            # transition | prior_consistent | out_of_range_near | out_of_range_far
    focus_organ: str
    focus_present: bool
    focus_area: int         # focus organ voxel count at z (for GT verification)
    present_organs: list    # all organs present at z (>= presence_min_pixels)
    boundary_organs: list   # organs at their z-boundary at this z
    queried: list = field(default_factory=list)  # [(organ, present_bool), ...]


def _pick(rng, pool: list[str], k: int, exclude: set[str]) -> list[str]:
    choices = [o for o in pool if o not in exclude]
    if not choices:
        return []
    k = min(k, len(choices))
    idx = rng.choice(len(choices), size=k, replace=False)
    return [choices[i] for i in idx]


def build_query_set(sample: SliceSample, areas, ranges, cfg, rng) -> list[tuple[str, bool]]:
    """Balanced ~n_present present + ~n_absent absent organs, always including the
    focus organ, and biased to include at least one boundary organ."""
    present = sorted(sample.present_organs)
    absent = [o for o in VOCAB if o not in sample.present_organs]
    chosen: list[tuple[str, bool]] = [(sample.focus_organ, sample.focus_present)]
    used = {sample.focus_organ}

    # Try to seed one boundary organ (other than focus) to keep the set hard.
    boundary_other = [o for o in sample.boundary_organs if o not in used]
    if boundary_other:
        b = boundary_other[rng.integers(len(boundary_other))]
        chosen.append((b, b in sample.present_organs))
        used.add(b)

    need_present = cfg.n_query_present - sum(1 for _, p in chosen if p)
    need_absent = cfg.n_query_absent - sum(1 for _, p in chosen if not p)
    for o in _pick(rng, present, max(need_present, 0), used):
        chosen.append((o, True)); used.add(o)
    for o in _pick(rng, absent, max(need_absent, 0), used):
        chosen.append((o, False)); used.add(o)

    rng.shuffle(chosen)
    return chosen


def sample_volume(case: str, label_vol: np.ndarray, cfg: Config, rng) -> list[SliceSample]:
    areas = organ_areas(label_vol)
    ranges = z_ranges(areas, cfg.presence_min_pixels)
    n_z = label_vol.shape[2]
    thr = cfg.presence_min_pixels
    hard0 = cfg.near_miss_hard_zero
    samples: list[SliceSample] = []

    def mk(z, stratum, focus):
        pres = sorted(present_at_z(areas, z, thr))
        return SliceSample(
            case=case, z=int(z), stratum=stratum, focus_organ=focus,
            focus_present=focus in pres, focus_area=int(areas[focus][z]),
            present_organs=pres,
            boundary_organs=sorted(boundary_organs_at_z(ranges, z)),
        )

    for focus in cfg.focus_organs:
        if focus not in ranges:
            continue
        zmin, zmax = ranges[focus]

        # transition: appearance + disappearance slices (focus marginally present)
        for z in {zmin, zmax}:
            samples.append(mk(z, "transition", focus))

        # prior_consistent: n slices evenly across the organ's mid-range (knob 1).
        # Require the focus to be solidly present (>= thr) at each.
        seen_pc = set()
        for frac in np.linspace(0.25, 0.75, cfg.n_prior_consistent):
            z = int(round(zmin + frac * (zmax - zmin)))
            if z not in seen_pc and areas[focus][z] >= thr:
                seen_pc.add(z)
                samples.append(mk(z, "prior_consistent", focus))

        # out_of_range_near: the near-miss negative just past each boundary
        # (knob 2). Scan offsets 2..4 outward; take the first slice that is a
        # HARD zero (< hard0 voxels) so partial-volume flicker can't sneak a
        # true-present in as a negative (knob 4).
        for boundary, direction in ((zmin, -1), (zmax, +1)):
            for k in range(cfg.near_miss_min_offset, cfg.near_miss_max_offset + 1):
                z = boundary + direction * k
                if 0 <= z < n_z and areas[focus][z] < hard0:
                    samples.append(mk(z, "out_of_range_near", focus))
                    break

        # out_of_range_far: well outside the range, the slice with the most OTHER
        # organs present; focus must be a hard zero there too.
        inf_region = list(range(0, max(zmin - cfg.out_of_range_margin, 0)))
        sup_region = list(range(zmax + cfg.out_of_range_margin, n_z))
        for region in (inf_region, sup_region):
            cand = [(len(present_at_z(areas, z, thr)), z)
                    for z in region if areas[focus][z] < hard0]
            cand = [(n, z) for n, z in cand if n > 0]
            if cand:
                _, z = max(cand)
                samples.append(mk(z, "out_of_range_far", focus))

    for s in samples:
        s.queried = build_query_set(s, areas, ranges, cfg, rng)
    return samples


# --------------------------------------------------------------------------- #
# Debug rendering
# --------------------------------------------------------------------------- #

def render_slice(vol: np.ndarray, z: int, cfg: Config) -> np.ndarray:
    gray = to_radiological(window_to_uint8(vol[:, :, z], cfg.window_center, cfg.window_width))
    return to_rgb(gray)


def overlay_focus(rgb: np.ndarray, label_vol: np.ndarray, z: int, focus: str) -> np.ndarray:
    """Green outline of the focus organ (if present) for GT verification."""
    out = rgb.copy()
    mask = to_radiological(label_vol[:, :, z] == NAME_TO_ID[focus]).astype(bool)
    if mask.sum():
        ring = mask & ~binary_erosion(mask, iterations=2)
        out[ring] = (0, 255, 0)
    return out


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #

def dry_run(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    images = sorted(Path(cfg.images_dir).glob("*.nii*"))
    labels = sorted(Path(cfg.labels_dir).glob("*.nii*"))
    if not images or len(images) != len(labels):
        raise SystemExit(f"image/label mismatch: {len(images)} images, {len(labels)} labels")
    images, labels = images[: cfg.max_cases], labels[: cfg.max_cases]

    dbg = Path(cfg.out_dir) / "debug_strata"
    if cfg.save_debug_images:
        dbg.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    strata_order = ["transition", "prior_consistent", "out_of_range_near", "out_of_range_far"]
    negative_strata = {"out_of_range_near", "out_of_range_far"}
    strata_counts = {k: 0 for k in strata_order}
    flicker_violations = []   # negative slices where the focus is NOT a hard zero
    all_samples = []

    print(f"{'case':9s} {'z':>3s} {'stratum':17s} {'focus':12s} {'fGT':3s} "
          f"{'farea':>5s} {'#pre':>4s}  present_organs | queried(P/A, *=boundary)")
    print("-" * 150)

    for img_path, lbl_path in zip(images, labels):
        vol = load_volume(str(img_path))
        lbl = load_volume(str(lbl_path))
        case = img_path.name.split(".")[0]
        samples = sample_volume(case, lbl, cfg, rng)
        samples.sort(key=lambda s: (s.focus_organ, s.z))
        all_samples.extend(samples)

        for s in samples:
            strata_counts[s.stratum] += 1
            if s.stratum in negative_strata and s.focus_area >= cfg.near_miss_hard_zero:
                flicker_violations.append(s)
            q = " ".join(
                f"{o}:{'P' if p else 'A'}{'*' if o in s.boundary_organs else ''}"
                for o, p in s.queried)
            pres = ",".join(s.present_organs) if s.present_organs else "(none)"
            print(f"{s.case:9s} {s.z:3d} {s.stratum:17s} {s.focus_organ:12s} "
                  f"{'Y' if s.focus_present else 'N':3s} {s.focus_area:5d} "
                  f"{len(s.present_organs):4d}  {pres[:46]:46s} | {q}")

            if cfg.save_debug_images:
                rgb = render_slice(vol, s.z, cfg)
                if s.focus_present:
                    rgb = overlay_focus(rgb, lbl, s.z, s.focus_organ)
                fn = f"{case}_z{s.z:03d}_{s.stratum}_{s.focus_organ}.png"
                Image.fromarray(rgb).save(dbg / fn)

    print("-" * 150)
    total = sum(strata_counts.values())
    print(f"TOTAL samples: {total}  |  " +
          "  ".join(f"{k}={v}" for k, v in strata_counts.items()))

    # Balance + GT integrity checks
    n_pos = sum(1 for s in all_samples for o, p in s.queried if p)
    n_neg = sum(1 for s in all_samples for o, p in s.queried if not p)
    print(f"queried items: present={n_pos} absent={n_neg}")
    print(f"GT CHECK: negative-stratum slices with focus_area >= {cfg.near_miss_hard_zero} "
          f"(should be 0): {len(flicker_violations)}")
    for s in flicker_violations:
        print(f"   !! {s.case} z={s.z} {s.stratum} {s.focus_organ} area={s.focus_area}")
    if cfg.save_debug_images:
        print(f"debug PNGs -> {dbg}")
    print("\nSTEP 1 ONLY — no model calls. Eyeball the table + PNGs before step 2.")


# --------------------------------------------------------------------------- #
# STEP 2: prompts, model wrapper, parsing, scoring, sweep
# --------------------------------------------------------------------------- #

FORMATS = ["freetext", "verdict_first", "reasoning_first"]
JSON_FORMATS = {"verdict_first", "reasoning_first"}

# organ -> substrings that identify it in model text / JSON keys. Side-specific
# kidney/adrenal synonyms avoid cross-matching left vs right.
ORGAN_SYNONYMS = {
    "spleen": ["spleen", "splenic"],
    "right_kidney": ["right kidney", "right renal"],
    "left_kidney": ["left kidney", "left renal"],
    "gallbladder": ["gallbladder", "gall bladder"],
    "esophagus": ["esophagus", "oesophagus"],
    "liver": ["liver", "hepatic"],
    "stomach": ["stomach", "gastric"],
    "aorta": ["aorta", "aortic"],
    "inferior_vena_cava": ["inferior vena cava", "ivc", "vena cava"],
    "portal_splenic_vein": ["portal", "splenic vein"],
    "pancreas": ["pancreas", "pancreatic"],
    "right_adrenal_gland": ["right adrenal"],
    "left_adrenal_gland": ["left adrenal"],
}


def _disp(organ: str) -> str:
    return organ.replace("_", " ")


def _norm(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def match_organ(text: str, organ: str) -> bool:
    t = _norm(text)
    return any(s in t for s in ORGAN_SYNONYMS[organ])


def build_prompt(fmt: str, queried: list[tuple[str, bool]]) -> str:
    """Prompt for one format. Queried order is the pre-shuffled, recorded order;
    the JSON presence map keys are EXACTLY the queried organs."""
    qnames = [_disp(o) for o, _ in queried]
    qlist = ", ".join(qnames)
    vlist = ", ".join(_disp(o) for o in VOCAB)
    head = ("This is one axial abdominal CT slice.\n"
            f"Organs segmented in this dataset: {vlist}.\n")

    if fmt == "freetext":
        return (head +
                f"For THIS slice, consider only these organs: {qlist}.\n"
                "List which of them are actually present in this slice. Reply "
                "with organ names only, comma-separated. If none of them are "
                "present, reply exactly 'none'.")

    keys = ",\n".join(f'    "{n}": "present" or "absent"' for n in qnames)
    decide = (f"For THIS slice, decide for EACH of these organs whether it is "
              f"present or absent: {qlist}.\n")
    if fmt == "verdict_first":
        return (head + decide +
                "Respond strictly in JSON with exactly these keys and no others:\n"
                "{\n" + keys + "\n}")
    # reasoning_first: reasoning fields strictly BEFORE the presence map
    return (head + decide +
            "Respond strictly in JSON. Fill the fields IN ORDER; complete "
            "observation and reasoning BEFORE the presence map:\n"
            "{\n"
            '  "observation": "<brief description of the anatomy visible in this slice>",\n'
            '  "reasoning": "<for each listed organ, state whether it appears in this slice and why>",\n'
            '  "presence": {\n' + keys + "\n  }\n"
            "}")


def _extract_json(raw: str):
    blob = raw.strip()
    if "```" in blob:
        blob = re.sub(r"```(?:json)?", "", blob)
    m = re.search(r"\{.*\}", blob, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _presence_value(pmap: dict, organ: str):
    """Return 'present'/'absent'/None(=unrecognised) by matching a map key."""
    for k, v in pmap.items():
        if match_organ(k, organ):
            s = str(v).strip().lower()
            if "present" in s or s in ("yes", "true", "1"):
                return "present"
            if "absent" in s or s in ("no", "false", "0", "not present"):
                return "absent"
            return None
    return "MISSING"


def parse_response(raw: str, fmt: str, queried: list[tuple[str, bool]]) -> dict:
    """organ -> {pred_present: bool|None, parse_fail: bool}. A missing JSON key
    or an unparseable blob is parse_fail, NEVER silently 'absent'."""
    organs = [o for o, _ in queried]
    if fmt == "freetext":
        text = _norm(raw)
        if not text:
            return {o: {"pred_present": None, "parse_fail": True} for o in organs}
        is_none = text == "none" or text.startswith("none")
        out = {}
        for o in organs:
            present = False if is_none else match_organ(raw, o)
            out[o] = {"pred_present": present, "parse_fail": False}
        return out

    obj = _extract_json(raw)
    if obj is None:
        return {o: {"pred_present": None, "parse_fail": True} for o in organs}
    pmap = obj.get("presence") if fmt == "reasoning_first" and isinstance(obj.get("presence"), dict) else obj
    if not isinstance(pmap, dict):
        return {o: {"pred_present": None, "parse_fail": True} for o in organs}

    out = {}
    for o in organs:
        val = _presence_value(pmap, o)
        if val in ("MISSING", None):          # missing key OR unrecognised value
            out[o] = {"pred_present": None, "parse_fail": True}
        else:
            out[o] = {"pred_present": val == "present", "parse_fail": False}
    return out


class MedGemma:
    def __init__(self, cfg: Config):
        import torch
        from scope.models.medgemma_native import load_model
        self.torch = torch
        self.cfg = cfg
        self.model, self.processor = load_model(model_id=cfg.model_id)
        self.model.eval()
        self._pan = cfg.do_pan_and_scan
        dev = getattr(self.model, "hf_device_map", None) or next(self.model.parameters()).device
        print(f"[model] {cfg.model_id}\n[model] device map: {dev}", file=sys.stderr)

    def ask(self, img_rgb, prompt: str) -> str:
        from PIL import Image
        torch = self.torch
        if img_rgb is None:                       # text-only (no-image control)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            kwargs = {}
        else:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(img_rgb).convert("RGB")},
                {"type": "text", "text": prompt},
            ]}]
            kwargs = {"do_pan_and_scan": True} if self._pan else {}
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", **kwargs)
        except (TypeError, ValueError):
            self._pan = False
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        ilen = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens,
                                      do_sample=False)
        return self.processor.decode(gen[0][ilen:], skip_special_tokens=True).strip()


def _sample_id(s: SliceSample) -> str:
    return f"{s.case}_z{s.z:03d}_{s.stratum}_{s.focus_organ}"


def run_sweep(cfg: Config) -> None:
    import time
    rng = np.random.default_rng(cfg.seed)
    images = sorted(Path(cfg.images_dir).glob("*.nii*"))
    labels = sorted(Path(cfg.labels_dir).glob("*.nii*"))
    if not images or len(images) != len(labels):
        raise SystemExit(f"image/label mismatch: {len(images)} images, {len(labels)} labels")
    images, labels = images[: cfg.max_cases], labels[: cfg.max_cases]

    # Build all samples first (cheap, no model) so the queried set is fixed once
    # per slice and the full-sweep size is known for the wall-clock estimate.
    flat: list[tuple[str, str, SliceSample]] = []
    for img_path, lbl_path in zip(images, labels):
        lblv = load_volume(str(lbl_path))
        case = img_path.name.split(".")[0]
        for s in sample_volume(case, lblv, cfg, rng):
            flat.append((str(img_path), case, s))
    full_n = len(flat)
    run = flat[: cfg.limit_n] if cfg.limit_n else flat
    print(f"[sweep] {full_n} slices total; running {len(run)} "
          f"x {len(FORMATS)} formats = {len(run)*len(FORMATS)} calls", file=sys.stderr)

    model = MedGemma(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    call_times = []
    volcache: dict[str, np.ndarray] = {}
    for imgp, case, s in run:
        if imgp not in volcache:
            volcache[imgp] = load_volume(imgp)
        rgb = render_slice(volcache[imgp], s.z, cfg)
        sid = _sample_id(s)
        for fmt in FORMATS:
            prompt = build_prompt(fmt, s.queried)
            t0 = time.perf_counter()
            raw = model.ask(rgb, prompt)
            dt = time.perf_counter() - t0
            call_times.append(dt)
            if len(call_times) == 1:
                est = dt * full_n * len(FORMATS)
                print(f"[timing] first call {dt:.1f}s (incl. warmup) -> full sweep "
                      f"{full_n*len(FORMATS)} calls ~ {est/3600:.1f} h", file=sys.stderr)
            parsed = parse_response(raw, fmt, s.queried)
            for organ, gt in s.queried:
                p = parsed[organ]
                correct = (not p["parse_fail"]) and (p["pred_present"] == gt)
                records.append({
                    "sample_id": sid, "case": case, "z": s.z, "stratum": s.stratum,
                    "focus_organ": s.focus_organ, "focus_area": s.focus_area,
                    "queried_set": [o for o, _ in s.queried],
                    "organ": organ, "ground_truth_present": bool(gt),
                    "format": fmt, "raw": raw,
                    "parsed_present": p["pred_present"], "parse_fail": p["parse_fail"],
                    "correct": bool(correct),
                })

    if call_times:
        med = float(np.median(call_times))
        print(f"[timing] {len(call_times)} calls, median {med:.1f}s -> full sweep "
              f"{full_n*len(FORMATS)} calls ~ {med*full_n*len(FORMATS)/3600:.1f} h",
              file=sys.stderr)

    tag = f"smoke_limit{cfg.limit_n}" if cfg.limit_n else "results"
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config_seed": cfg.seed, "full_n_slices": full_n,
                   "median_call_s": float(np.median(call_times)) if call_times else None,
                   "records": records}, f, indent=2)
    print(f"[done] {len(records)} records -> {out_path}", file=sys.stderr)

    if cfg.limit_n:
        _print_smoke(run, records)
    else:
        analyze(records, out_dir, n_boot=cfg.n_boot, seed=cfg.seed)


def _print_smoke(run, records) -> None:
    by_sid_fmt = {}
    for r in records:
        by_sid_fmt.setdefault((r["sample_id"], r["format"]), []).append(r)
    pf = {fmt: [0, 0] for fmt in FORMATS}
    for _, _, s in run:
        sid = _sample_id(s)
        print("\n" + "=" * 100)
        print(f"{sid}  | stratum={s.stratum} focus={s.focus_organ}(area={s.focus_area}) "
              f"GT: " + ", ".join(f"{o}={'P' if g else 'A'}" for o, g in s.queried))
        for fmt in FORMATS:
            rs = by_sid_fmt[(sid, fmt)]
            raw = rs[0]["raw"].replace("\n", " ")
            print(f"\n  [{fmt}] raw: {raw[:220]}")
            cells = []
            for r in rs:
                pf[fmt][1] += 1
                if r["parse_fail"]:
                    pf[fmt][0] += 1
                    mark = "PARSE_FAIL"
                else:
                    mark = ("present" if r["parsed_present"] else "absent") + \
                           (" OK" if r["correct"] else " WRONG")
                cells.append(f"{r['organ']}[gt={'P' if r['ground_truth_present'] else 'A'}]->{mark}")
            print("    " + " | ".join(cells))
    print("\n" + "=" * 100)
    print("parse-fail rate per format:")
    for fmt in FORMATS:
        n_fail, n = pf[fmt]
        print(f"  {fmt:16s} {n_fail}/{n} ({n_fail/n:.2f})" if n else f"  {fmt}: n=0")
    print("\nSMOKE TEST — eyeball raw+parsed+GT alignment before the full sweep.")


# --------------------------------------------------------------------------- #
# STEP 3: scoring, McNemar, patient-clustered bootstrap
# --------------------------------------------------------------------------- #
#
# Primary endpoint: NEAR-MISS SPECIFICITY (out_of_range_near). It is the only
# metric a blanket-"present" model cannot fake: every near-miss item is truly
# absent, so "present" there is an unambiguous z-prior fabrication. Balanced
# accuracy is deliberately NOT the headline — it is inflated by the present-bias
# on positive-heavy strata. The far-vs-near specificity gap per format is the
# real result. Every accuracy is printed with its base rate (fraction truly
# present) and the always-present baseline so the present-bias stays visible.

STRATA_ORDER = ["prior_consistent", "transition", "out_of_range_near", "out_of_range_far"]


def _sensitivity(items):
    pos = [r for r in items if r["ground_truth_present"] and not r["parse_fail"]]
    return (sum(r["parsed_present"] for r in pos) / len(pos), len(pos)) if pos else (float("nan"), 0)


def _specificity(items):
    neg = [r for r in items if (not r["ground_truth_present"]) and not r["parse_fail"]]
    return (sum(not r["parsed_present"] for r in neg) / len(neg), len(neg)) if neg else (float("nan"), 0)


def _base_rate(items):
    return (sum(r["ground_truth_present"] for r in items) / len(items), len(items)) if items else (float("nan"), 0)


def _parse_fail_rate(items):
    return (sum(r["parse_fail"] for r in items) / len(items), len(items)) if items else (float("nan"), 0)


def _bootstrap_ci(items, metric_fn, n_boot, seed):
    """Patient-clustered bootstrap: resample CASES (not slices) with replacement."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_case = defaultdict(list)
    for r in items:
        by_case[r["case"]].append(r)
    cl = list(by_case)
    if len(cl) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cl), len(cl))
        samp = [r for i in idx for r in by_case[cl[i]]]
        v, _ = metric_fn(samp)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def _mcnemar(items_a, items_b):
    """Paired McNemar on (sample_id, organ) correctness; parsed items only."""
    import scipy.stats as st
    a = {(r["sample_id"], r["organ"]): r["correct"] for r in items_a if not r["parse_fail"]}
    b = {(r["sample_id"], r["organ"]): r["correct"] for r in items_b if not r["parse_fail"]}
    bc = cc = 0
    for k in set(a) & set(b):
        if a[k] and not b[k]:
            bc += 1
        elif b[k] and not a[k]:
            cc += 1
    n = bc + cc
    p = st.binomtest(min(bc, cc), n, 0.5).pvalue if n > 0 else float("nan")
    return bc, cc, p


def _r(v):
    return "  nan" if v != v else f"{v:.3f}"


def analyze(records: list, out_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    cases = sorted({r["case"] for r in records})
    organs = sorted({r["organ"] for r in records})

    def sel(fmt=None, stratum=None, organ=None):
        return [r for r in records
                if (fmt is None or r["format"] == fmt)
                and (stratum is None or r["stratum"] == stratum)
                and (organ is None or r["organ"] == organ)]

    print("\n" + "#" * 90)
    print(f"# ANALYSIS  cases={len(cases)}  records={len(records)}  bootstrap={n_boot} (clustered on case)")
    print("#" * 90)

    # --- base rates + flat (always-present) baseline ---
    print("\n=== BASE RATES & FLAT BASELINE (always-present model) ===")
    print(f"{'stratum':18s} {'n':>5s} {'frac_present':>12s}   always-present: sens / spec / acc")
    for st_ in STRATA_ORDER:
        br, n = _base_rate(sel(stratum=st_))
        # always-present: sens=1, spec=0, acc=frac_present
        print(f"{st_:18s} {n:5d} {_r(br):>12s}   1.000 / 0.000 / {_r(br)}")

    # --- PRIMARY: near-miss specificity ---
    print("\n=== PRIMARY ENDPOINT — near-miss specificity (always-present baseline = 0.000) ===")
    print(f"{'format':16s} {'spec':>6s}  {'95% CI (case-clustered)':>24s}  {'n_neg':>5s}  {'parse_fail':>10s}")
    summary = {"near_miss_specificity": {}, "far_specificity": {}, "far_minus_near_gap": {}}
    for fmt in FORMATS:
        items = sel(fmt=fmt, stratum="out_of_range_near")
        sp, n = _specificity(items)
        lo, hi = _bootstrap_ci(items, _specificity, n_boot, seed)
        pf, _ = _parse_fail_rate(items)
        summary["near_miss_specificity"][fmt] = {"spec": sp, "ci": [lo, hi], "n": n, "parse_fail": pf}
        print(f"{fmt:16s} {_r(sp):>6s}  [{_r(lo)}, {_r(hi)}]{'':>6s}  {n:5d}  {_r(pf):>10s}")

    # --- far specificity (context) ---
    print("\n=== far specificity (context; easy negatives) ===")
    print(f"{'format':16s} {'spec':>6s}  {'95% CI':>24s}  {'n_neg':>5s}")
    for fmt in FORMATS:
        items = sel(fmt=fmt, stratum="out_of_range_far")
        sp, n = _specificity(items)
        lo, hi = _bootstrap_ci(items, _specificity, n_boot, seed)
        summary["far_specificity"][fmt] = {"spec": sp, "ci": [lo, hi], "n": n}
        print(f"{fmt:16s} {_r(sp):>6s}  [{_r(lo)}, {_r(hi)}]{'':>6s}  {n:5d}")

    # --- far vs near gap (the real result) ---
    print("\n=== FAR - NEAR specificity gap per format (grounding margin; bootstrap CI on gap) ===")
    print(f"{'format':16s} {'far':>6s} {'near':>6s} {'gap':>7s}  {'95% CI (gap)':>22s}")
    for fmt in FORMATS:
        far = sel(fmt=fmt, stratum="out_of_range_far")
        near = sel(fmt=fmt, stratum="out_of_range_near")
        fsp, _ = _specificity(far)
        nsp, _ = _specificity(near)

        def gap_metric(items, _fmt=fmt):
            f = _specificity([r for r in items if r["stratum"] == "out_of_range_far"])[0]
            nr = _specificity([r for r in items if r["stratum"] == "out_of_range_near"])[0]
            return ((f - nr) if (f == f and nr == nr) else float("nan"), 0)
        lo, hi = _bootstrap_ci(far + near, gap_metric, n_boot, seed)
        gap = fsp - nsp if (fsp == fsp and nsp == nsp) else float("nan")
        summary["far_minus_near_gap"][fmt] = {"gap": gap, "ci": [lo, hi]}
        print(f"{fmt:16s} {_r(fsp):>6s} {_r(nsp):>6s} {_r(gap):>7s}  [{_r(lo)}, {_r(hi)}]")

    # --- sensitivity by stratum (context, inflated by present-bias) ---
    print("\n=== sensitivity by stratum x format (context; base_rate shown) ===")
    print(f"{'stratum':18s} {'format':16s} {'sens':>6s} {'n_pos':>5s}")
    for st_ in ["prior_consistent", "transition"]:
        for fmt in FORMATS:
            se, n = _sensitivity(sel(fmt=fmt, stratum=st_))
            print(f"{st_:18s} {fmt:16s} {_r(se):>6s} {n:5d}")

    # --- per-organ near-miss specificity (format x organ) ---
    print("\n=== per-organ near-miss specificity (format x organ; n in parens) ===")
    print(f"{'organ':20s} " + " ".join(f"{f[:12]:>14s}" for f in FORMATS))
    per_organ_spec = {}
    for o in organs:
        cells = []
        row = {}
        for fmt in FORMATS:
            sp, n = _specificity(sel(fmt=fmt, stratum="out_of_range_near", organ=o))
            row[fmt] = {"spec": sp, "n": n}
            cells.append(f"{_r(sp)}({n})")
        if any(row[f]["n"] for f in FORMATS):
            per_organ_spec[o] = row
            print(f"{o:20s} " + " ".join(f"{c:>14s}" for c in cells))
    summary["per_organ_near_miss_specificity"] = per_organ_spec

    # --- McNemar per organ per format-pair on near-miss (within organ, never pooled) ---
    print("\n=== McNemar on near-miss correctness — per organ, per format-pair (NEVER pooled) ===")
    print(f"{'organ':20s} {'pair':32s} {'b':>4s} {'c':>4s} {'p':>8s}")
    pairs = [("freetext", "verdict_first"), ("freetext", "reasoning_first"),
             ("verdict_first", "reasoning_first")]
    mcnemar_out = {}
    for o in organs:
        for fa, fb in pairs:
            ia = sel(fmt=fa, stratum="out_of_range_near", organ=o)
            ib = sel(fmt=fb, stratum="out_of_range_near", organ=o)
            bc, cc, p = _mcnemar(ia, ib)
            if bc + cc > 0:
                mcnemar_out[f"{o}|{fa}_vs_{fb}"] = {"b": bc, "c": cc, "p": p}
                print(f"{o:20s} {fa+' vs '+fb:32s} {bc:4d} {cc:4d} {_r(p):>8s}")

    summary["mcnemar_near_miss"] = mcnemar_out

    # --- parse-fail rate (format x stratum) ---
    print("\n=== parse-fail rate (format x stratum) ===")
    print(f"{'stratum':18s} " + " ".join(f"{f[:12]:>14s}" for f in FORMATS))
    for st_ in STRATA_ORDER:
        cells = [f"{_r(_parse_fail_rate(sel(fmt=f, stratum=st_))[0])}" for f in FORMATS]
        print(f"{st_:18s} " + " ".join(f"{c:>14s}" for c in cells))

    out_path = Path(out_dir) / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[analysis] summary -> {out_path}")
    return summary


def analyze_file(path: str, n_boot: int, seed: int) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    analyze(data["records"], Path(path).parent, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", default=cfg.images_dir)
    ap.add_argument("--labels_dir", default=cfg.labels_dir)
    ap.add_argument("--out_dir", default=cfg.out_dir)
    ap.add_argument("--focus_organs", nargs="+", default=list(cfg.focus_organs))
    ap.add_argument("--presence_min_pixels", type=int, default=cfg.presence_min_pixels)
    ap.add_argument("--near_miss_hard_zero", type=int, default=cfg.near_miss_hard_zero)
    ap.add_argument("--n_prior_consistent", type=int, default=cfg.n_prior_consistent)
    ap.add_argument("--near_miss_min_offset", type=int, default=cfg.near_miss_min_offset)
    ap.add_argument("--near_miss_max_offset", type=int, default=cfg.near_miss_max_offset)
    ap.add_argument("--out_of_range_margin", type=int, default=cfg.out_of_range_margin)
    ap.add_argument("--max_cases", type=int, default=cfg.max_cases)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--no_debug_images", action="store_true")
    ap.add_argument("--model_id", default=cfg.model_id)
    ap.add_argument("--max_new_tokens", type=int, default=cfg.max_new_tokens)
    ap.add_argument("--no_pan_and_scan", action="store_true")
    ap.add_argument("--limit_n", type=int, default=cfg.limit_n,
                    help="smoke test: run only the first N slices end-to-end")
    ap.add_argument("--n_boot", type=int, default=cfg.n_boot)
    ap.add_argument("--dry_run", action="store_true",
                    help="step-1 strata table only (no model)")
    ap.add_argument("--analyze", default=None,
                    help="step-3 only: analyze an existing results JSON, no model")
    a = ap.parse_args()

    if a.analyze:
        cfg.n_boot = a.n_boot
        analyze_file(a.analyze, n_boot=a.n_boot, seed=a.seed)
        raise SystemExit(0)

    cfg.images_dir = a.images_dir
    cfg.labels_dir = a.labels_dir
    cfg.out_dir = a.out_dir
    cfg.focus_organs = tuple(a.focus_organs)
    cfg.presence_min_pixels = a.presence_min_pixels
    cfg.near_miss_hard_zero = a.near_miss_hard_zero
    cfg.n_prior_consistent = a.n_prior_consistent
    cfg.near_miss_min_offset = a.near_miss_min_offset
    cfg.near_miss_max_offset = a.near_miss_max_offset
    cfg.out_of_range_margin = a.out_of_range_margin
    cfg.max_cases = None if a.max_cases is not None and a.max_cases <= 0 else a.max_cases
    cfg.seed = a.seed
    cfg.save_debug_images = not a.no_debug_images
    cfg.model_id = a.model_id
    cfg.max_new_tokens = a.max_new_tokens
    cfg.do_pan_and_scan = not a.no_pan_and_scan
    cfg.limit_n = a.limit_n
    cfg.n_boot = a.n_boot
    cfg.dry_run = a.dry_run
    return cfg


if __name__ == "__main__":
    _cfg = parse_args()
    if _cfg.dry_run:
        dry_run(_cfg)
    else:
        run_sweep(_cfg)
