"""
Step 0: Regenerate type-separated CVAT tracks into the repo.

The on-disk `type_sep/tracks.tiff` files are inconsistent across crops (some are
3D mixed, some are 4D (T,H,W,2)).  To get a uniform T-cell source (and the cancer
lineage graph) we regenerate them from each crop's `annotations.xml` using the
existing CLI `parseXMLgt.py --separate_types 1`, which writes:
    tracks.tiff   (T, H, W, 2)   channel 0 = non-cancer (T cells), channel 1 = cancer
    cancer_ids.pkl                list of cancer CVAT track_ids
    graph.pkl                     {child_track_id: parent_track_id} lineage

Outputs go to an in-repo directory: {type_sep_tracks_dir}/{crop}/

Usage (occident):
    conda run -n occident python regen_type_sep_tracks.py
"""

import os
import re
import sys
import subprocess
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
cvat_base_dir = cfg["cvat_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
parsexml_repo = cfg["parsexml_repo"]
thresh = cfg.get("parsexml_thresh", 0.7)

parsexml_script = os.path.join(parsexml_repo, "scripts", "cli_scripts", "parseXMLgt.py")


def find_latest_xml(crop_dir):
    """Return the most recently modified `.xml` directly in `crop_dir`.

    Each crop directory can hold several annotation exports (e.g.
    `annotations.xml`, `annotations_50fv4.xml`); `annotations.xml` is not always
    the newest.  We pick the most recent by modification time and intentionally
    ignore nested dirs like `intermediate_results/` (non-recursive glob).
    """
    xmls = sorted(Path(crop_dir).glob("*.xml"), key=lambda p: p.stat().st_mtime)
    return str(xmls[-1]) if xmls else None


def parse_crop_hw(crop_id):
    """Derive (image_height, image_width) from a crop_id.

    crop_id looks like `B4_t50t100y200y350x750x900`; the part after the first
    underscore encodes t{start}t{end}y{start}y{end}x{start}x{end}.  Height is the
    y-span, width is the x-span.
    """
    slice_part = crop_id.split("_", 1)[1]
    matches = re.findall(r"([a-zA-Z])(\d+)", slice_part)
    vals = {}
    # ordered: t,t,y,y,x,x (,s)
    keys = ["t0", "t1", "y0", "y1", "x0", "x1", "s"]
    for (letter, value), key in zip(matches, keys):
        vals[key] = int(value)
    height = vals["y1"] - vals["y0"]
    width = vals["x1"] - vals["x0"]
    return height, width


print("=" * 60)
print("Step 0: Regenerating type-separated tracks")
print("=" * 60)
print(f"parseXMLgt.py: {parsexml_script}")
if not os.path.exists(parsexml_script):
    sys.exit(f"ERROR: parseXMLgt.py not found at {parsexml_script}")

for crop in crop_ids:
    well = crop.split("_")[0]
    crop_dir = os.path.join(cvat_base_dir, well, crop)
    xml_file = find_latest_xml(crop_dir)
    out_dir = os.path.join(type_sep_tracks_dir, crop)
    height, width = parse_crop_hw(crop)

    print("\n" + "-" * 60)
    print(f"Crop {crop}: H={height}, W={width}")
    print(f"  xml:  {xml_file}")
    print(f"  out:  {out_dir}")

    if xml_file is None:
        sys.exit(f"ERROR: no .xml file found in {crop_dir}")

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, parsexml_script,
        "-xml_file", xml_file,
        "-output_path", out_dir,
        "-image_height", str(height),
        "-image_width", str(width),
        "--thresh", str(thresh),
        "--separate_types", "1",
    ]
    # parseXMLgt.py does `from scripts.utils.XMLutils import ...`, so the
    # MarsonImagingPipeline repo root must be importable -> run with that cwd.
    env = dict(os.environ)
    env["PYTHONPATH"] = parsexml_repo + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=parsexml_repo, env=env)

print("\n" + "=" * 60)
print("Done regenerating type-separated tracks.")
print("=" * 60)
