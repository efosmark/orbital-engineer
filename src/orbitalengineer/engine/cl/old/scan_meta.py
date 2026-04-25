#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Dict, Any, List, Optional


def classify_vgpr(v: Optional[int]) -> str:
    if v is None:
        return "UNKNOWN"
    if v <= 32:
        return "OK"
    if v <= 64:
        return "WARN (moderate VGPR usage)"
    return "BAD (high VGPR usage, may hurt occupancy)"


def classify_sgpr(s: Optional[int]) -> str:
    if s is None:
        return "UNKNOWN"
    if s <= 64:
        return "OK"
    if s <= 96:
        return "WARN (could impact occupancy on some chips)"
    return "BAD (high SGPR usage, likely occupancy limiter)"


def overall_status(sgpr_status: str, vgpr_status: str) -> str:
    ranks = {"OK": 0, "WARN": 1, "BAD": 2, "UNKNOWN": 1}
    worst = max(ranks.get(sgpr_status.split()[0], 1),
                ranks.get(vgpr_status.split()[0], 1))
    if worst == 0:
        return "OK"
    if worst == 1:
        return "WARN"
    return "BAD"


def parse_meta_yaml_like(text: str) -> List[Dict[str, Any]]:
    """
    Try to parse the .meta as YAML using PyYAML if available.
    Returns a list of kernel dicts with at least: name, sgpr_count, vgpr_count.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        print(e)
        return []

    try:
        data = yaml.safe_load(text)
    except Exception as e:
        print(e)
        return []

    kernels_out: List[Dict[str, Any]] = []

    # V3-style: amdhsa.kernels: [ { .name: ..., .sgpr_count: ..., .vgpr_count: ... }, ... ]
    if isinstance(data, dict):
        candidates = data.get("amdhsa.kernels") or data.get("Kernels")
        if isinstance(candidates, list):
            for k in candidates:
                if not isinstance(k, dict):
                    continue
                name = (
                    k.get(".name")
                    or k.get("name")
                    or k.get("Name")
                    or k.get("SymbolName")
                )
                sgpr = (
                    k.get(".sgpr_count")
                    or k.get("sgpr_count")
                    or k.get("SGPRs")
                )
                vgpr = (
                    k.get(".vgpr_count")
                    or k.get("vgpr_count")
                    or k.get("VGPRs")
                )
                try:
                    sgpr_val = int(sgpr) if sgpr is not None else None
                except ValueError:
                    sgpr_val = None
                try:
                    vgpr_val = int(vgpr) if vgpr is not None else None
                except ValueError:
                    vgpr_val = None

                if name is None:
                    continue

                kernels_out.append(
                    {
                        "name": str(name),
                        "sgpr_count": sgpr_val,
                        "vgpr_count": vgpr_val,
                    }
                )

    return kernels_out


def parse_meta_regex(text: str) -> List[Dict[str, Any]]:
    """
    Fallback parser for .meta that looks like:

    amdhsa.kernels:
      - .name: my_kernel
        .sgpr_count: 18
        .vgpr_count: 10
      - .name: other_kernel
        .sgpr_count: 40
        .vgpr_count: 64
    """
    kernels: List[Dict[str, Any]] = []

    # Roughly split into per-kernel blocks based on "- .name:" lines
    # and also support "- Name:" / ".name:" variants.
    name_line_re = re.compile(
        r"^\s*-\s*(?:\.name|name|Name|SymbolName)\s*:\s*(\S+)"
    )
    sgpr_re = re.compile(
        r"\.(?:sgpr_count|num_sgprs)\s*:\s*(\d+)|\bSGPRs\s*:\s*(\d+)",
        re.IGNORECASE,
    )
    vgpr_re = re.compile(
        r"\.(?:vgpr_count|num_vgprs)\s*:\s*(\d+)|\bVGPRs\s*:\s*(\d+)",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    current_name = None
    current_block: List[str] = []

    def flush():
        nonlocal current_name, current_block, kernels
        if current_name is None:
            current_block = []
            return
        block_text = "\n".join(current_block)
        sgpr_val = None
        vgpr_val = None

        m_sgpr = sgpr_re.search(block_text)
        if m_sgpr:
            sgpr_val = int(next(g for g in m_sgpr.groups() if g is not None))

        m_vgpr = vgpr_re.search(block_text)
        if m_vgpr:
            vgpr_val = int(next(g for g in m_vgpr.groups() if g is not None))

        kernels.append(
            {
                "name": current_name,
                "sgpr_count": sgpr_val,
                "vgpr_count": vgpr_val,
            }
        )
        current_name = None
        current_block = []

    for line in lines:
        m = name_line_re.match(line)
        if m:
            # New kernel block starts
            flush()
            current_name = m.group(1)
            current_block = []
        elif current_name is not None:
            # Inside a kernel block
            # Stop if we hit an empty line separating blocks
            if line.strip() == "" and current_block:
                flush()
            else:
                current_block.append(line)

    # End of file
    flush()
    return kernels


def load_kernels_from_meta(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        text = f.read()

    # If this came from an ISA/obj dump, it may include
    # assembly directives like:
    #
    #   .amdgpu_metadata
    #   ---
    #   amdhsa.kernels:
    #     - .name: ...
    #   ...
    #   .end_amdgpu_metadata
    #
    # Trim down to just the YAML-ish metadata block.
    start_idx = text.find(".amdgpu_metadata")
    end_idx = text.find(".end_amdgpu_metadata")

    if start_idx != -1:
        # Skip the directive line itself
        nl = text.find("\n", start_idx)
        if nl != -1:
            start_idx = nl + 1
        else:
            # Weird file, just start after the marker token
            start_idx += len(".amdgpu_metadata")
    else:
        start_idx = 0

    if end_idx != -1:
        # Stop right before the .end_amdgpu_metadata directive
        trimmed = text[start_idx:end_idx].strip()
    else:
        trimmed = text[start_idx:].strip()

    # Now parse only the trimmed metadata chunk
    kernels = parse_meta_yaml_like(trimmed)
    if kernels:
        return kernels

    # Fallback to regex parser
    return parse_meta_regex(trimmed)



def main():
    parser = argparse.ArgumentParser(
        description="Scan AMD .meta file for sgpr/vgpr usage per kernel."
    )
    parser.add_argument("meta_file", help="Path to .meta file (amdgpu_metadata)")

    args = parser.parse_args()

    try:
        kernels = load_kernels_from_meta(args.meta_file)
    except FileNotFoundError:
        print(f"Error: file not found: {args.meta_file}", file=sys.stderr)
        sys.exit(1)

    if not kernels:
        print("No kernels found or unable to parse metadata.", file=sys.stderr)
        sys.exit(1)

    for k in kernels:
        name = k.get("name", "<unknown>")
        sgpr = k.get("sgpr_count")
        vgpr = k.get("vgpr_count")

        sgpr_status = classify_sgpr(sgpr)
        vgpr_status = classify_vgpr(vgpr)
        overall = overall_status(sgpr_status, vgpr_status)

        print(f"Kernel: {name}")
        print(f"  SGPRs: {sgpr if sgpr is not None else 'N/A'}  [{sgpr_status}]")
        print(f"  VGPRs: {vgpr if vgpr is not None else 'N/A'}  [{vgpr_status}]")
        print(f"  Overall: {overall}")
        print()


if __name__ == "__main__":
    main()
