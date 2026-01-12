import os
import glob
from pathlib import Path
import json

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from nibabel.orientations import aff2axcodes

from model.build import build_model

# ---------------------------------------------------------------------
# RAS-aware shape @ 1mm iso
# ---------------------------------------------------------------------

def ras_shape_spacing(img):
    hdr = img.header
    shape = img.shape
    zooms = hdr.get_zooms()
    if len(shape) < 3 or len(zooms) < 3:
        raise ValueError(f"Not enough spatial dims: shape={shape}, zooms={zooms}")

    axcodes = aff2axcodes(img.affine)
    if axcodes is None or len(axcodes) < 3:
        raise ValueError(f"Cannot determine orientation codes: {axcodes}")

    sx = sy = sz = np.nan
    shx = shy = shz = np.nan

    for dim in range(3):
        code = axcodes[dim]
        dim_len = int(shape[dim])
        dim_zoom = float(zooms[dim])

        if code in ("R", "L"):
            shx = dim_len
            sx = dim_zoom
        elif code in ("A", "P"):
            shy = dim_len
            sy = dim_zoom
        elif code in ("S", "I"):
            shz = dim_len
            sz = dim_zoom
        else:
            raise ValueError(f"Unknown orientation code {code} for dim {dim}")

    if any(np.isnan([sx, sy, sz, shx, shy, shz])):
        raise ValueError(f"Incomplete RAS mapping: sx={sx}, sy={sy}, sz={sz}, shx={shx}, shy={shy}, shz={shz}, axcodes={axcodes}")

    return (shx, shy, shz), (sx, sy, sz), axcodes

def shape_at_1mm_iso_ras(img):
    (shx, shy, shz), (sx, sy, sz), axcodes = ras_shape_spacing(img)
    size_mm_x = shx * sx
    size_mm_y = shy * sy
    size_mm_z = shz * sz
    new_shx = max(1, int(np.round(size_mm_x / 1.0)))
    new_shy = max(1, int(np.round(size_mm_y / 1.0)))
    new_shz = max(1, int(np.round(size_mm_z / 1.0)))
    return (new_shx, new_shy, new_shz), (size_mm_x, size_mm_y, size_mm_z), axcodes

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def floor_to_multiple(v, m=32):
    return (v // m) * m
def closest_shape_under_prod(dhw, max_prod=13800000, multiple=32, min_size=32):
    """
    Find the closest (D,H,W) <= dhw such that:
      - each dim is multiple of `multiple`
      - D*H*W <= max_prod
    """
    D, H, W = dhw

    # 1) floor each axis
    D = floor_to_multiple(D, multiple)
    H = floor_to_multiple(H, multiple)
    W = floor_to_multiple(W, multiple)

    # guard
    D = max(D, min_size)
    H = max(H, min_size)
    W = max(W, min_size)

    # 2) iteratively reduce smallest-penalty axis
    while D * H * W > max_prod:
        print(f"  - Reducing shape {(D,H,W)} with prod {D*H*W} > {max_prod}")
        penalties = {
            "D": 32 * H * W if D > min_size else float("inf"),
            "H": 32 * D * W if H > min_size else float("inf"),
            "W": 32 * D * H if W > min_size else float("inf"),
        }

        axis = min(penalties, key=penalties.get)

        if penalties[axis] == float("inf"):
            raise RuntimeError("Cannot reduce shape further without breaking min_size.")

        if axis == "D":
            D -= 32
        elif axis == "H":
            H -= 32
        else:
            W -= 32

    return (D, H, W)

def estimate_param_mem_gib(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30

def can_train_step(model, shape, device="cuda", amp=True):
    try:
        model.train().to(device)
        x = torch.randn(shape, device=device)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                y = model(x)
                loss = y.float().square().mean()
        else:
            y = model(x)
            loss = y.square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        del x, y, loss
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False

def clamp_xyz(xyz, max_xyz):
    x, y, z = xyz
    mx, my, mz = max_xyz
    return (min(x, mx), min(y, my), min(z, mz))

# ---------------------------------------------------------------------
# Main: scan shapes (1mm iso proxy) then test unique clamped shapes for OOM
# ---------------------------------------------------------------------

def test_unique_shapes_for_oom_with_truncation(model, root_path_abs, in_channels=1, multiple=32, max_x=320, max_y=192, max_z=480, amp=True, device="cuda", verbose=False):
    discovered_folders = [
        os.path.join(root_path_abs, d)
        for d in os.listdir(root_path_abs)
        if os.path.isdir(os.path.join(root_path_abs, d))
    ]
    if not discovered_folders:
        raise RuntimeError(f"No dataset sub-folders found inside: {root_path_abs}")

    shape_to_files = {}
    total_files = 0
    truncated_files = []

    for folder in discovered_folders:
        pattern = os.path.join(folder, "sub-*", "**", "anat", "*.nii.gz")
        found_images = sorted(glob.glob(pattern, recursive=True))

        valid_images = [
            f for f in found_images
            if "preproc" not in f.lower()
            and ("lowres" not in f.lower() or Path(f.replace("lowres", "highres")).exists())
        ]

        for fpath in tqdm(valid_images, desc=f"Indexing {os.path.basename(folder.rstrip('/'))}", leave=False):
            total_files += 1
            try:
                img = nib.load(fpath)
                (x, y, z), size_mm_ras, axcodes = shape_at_1mm_iso_ras(img)  # x,y,z in logical RAS
                padded_shape = closest_shape_under_prod((x, y, z))
                if padded_shape != (x, y, z):
                    truncated_files.append({
                        "file": fpath,
                        "shape_1mm_ras": (x, y, z),
                        "shape_truncated_ras": padded_shape,
                        "size_mm_ras": tuple(float(v) for v in size_mm_ras),
                        "axcodes": tuple(axcodes),
                        "orig_shape": tuple(int(v) for v in img.shape[:3]),
                        "orig_zooms": tuple(float(v) for v in img.header.get_zooms()[:3]),
                    })
                    if verbose:
                        print(f"[TRUNC] {fpath} | 1mm(RAS)={(x,y,z)} -> trunc={(x2,y2,z2)} | axcodes={axcodes}")

                
                shape_to_files.setdefault(padded_shape, []).append(fpath)

            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

    unique_shapes = sorted(shape_to_files.keys())
    print(f"\nShapes: {len(unique_shapes)} unique padded resolutions (after 1mm-iso proxy + truncation).")
    for s in unique_shapes:
        print(" ", s, "voxels =", s[0] * s[1] * s[2], "n_files =", len(shape_to_files[s]))

    oom_shapes = []
    ok_shapes = []
    for (D, H, W) in unique_shapes:
        print(f"Testing shape {(D,H,W)} ...", end=" ", flush=True)
        ok = can_train_step(model, (1, in_channels, D, H, W), device=device, amp=amp)
        if ok:
            print("OK")
            ok_shapes.append((D, H, W))
        else:
            print("OOM")
            oom_shapes.append((D, H, W))

    oom_files = []
    for s in oom_shapes:
        oom_files.extend(shape_to_files.get(s, []))

    return {
        "shape_to_files": shape_to_files,
        "unique_shapes": unique_shapes,
        "ok_shapes": ok_shapes,
        "oom_shapes": oom_shapes,
        "oom_files": oom_files,
        "truncated_files": truncated_files,
        "total_files": total_files,
        "multiple": multiple,
    }

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

file = "/home/ge.polymtl.ca/p123239/SpineFoundation/model/SwinUNETR.json"
with open(file, "r") as f:
    config = json.load(f)
config.pop("model_name", None)
config.pop("img_resolution", None)

device = "cuda"
model = build_model("swin_unetr", params=config).to(device)

print("Model params GiB:", estimate_param_mem_gib(model))

root_path_abs = "/home/ge.polymtl.ca/p123239/data_sub"
res = test_unique_shapes_for_oom_with_truncation(
    model,
    root_path_abs,
    in_channels=1,
    multiple=32,
    max_x=160,
    max_y=192,
    max_z=480,
    amp=True,
    device=device,
    verbose=False,
)

out_json = "oom_shapes_and_files_after_1mm_iso_trunc_xyz.json"
with open(out_json, "w") as f:
    json.dump(
        {
            "oom_shapes": res["oom_shapes"],
            "ok_shapes": res["ok_shapes"],
            "oom_files": res["oom_files"],
            "truncated_files": res["truncated_files"],
            "limits_xyz": res["limits_xyz"],
            "multiple": res["multiple"],
        },
        f,
        indent=2,
    )
print(f"\nWrote: {out_json} | oom_shapes={len(res['oom_shapes'])} | oom_files={len(res['oom_files'])} | truncated_files={len(res['truncated_files'])}/{res['total_files']}")
