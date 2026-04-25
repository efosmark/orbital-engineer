import re
from collections import Counter


def slice_asm(asm_text, start=r"\.LBB0_6:", end=r"\.LBB0_28:"):
    s = re.search(start, asm_text)
    e = re.search(end, asm_text)
    return asm_text[s.start():e.start()] if s and e else asm_text


# --- ANSI colors (disable if you don't want color) ---
C = {
    "reset": "\033[0m",
    "op": "\033[1;36m",        # cyan bold for SIMD ops
    "reg": "\033[1;35m",       # magenta bold for xmm/ymm/zmm
    "width": "\033[1;33m",     # yellow bold for width badges
    "title": "\033[1;37m",     # white bold
}

# Common SIMD mnemonics; extend as needed
# change in SIMD_OPS pattern definition:
SIMD_OPS = r"""
v?(?:addp[sd]|subp[sd]|mulp[sd]|divp[sd]|maxp[sd]|minp[sd])|
v?(?:fmadd|fnmadd|fmsub|fnmsub)[123]?(?:132|213|231)?p[sd]|
v?(?:andp[sd]?|orp[sd]?|xorp[sd]?)|
v?(?:sh ufp[sd]?|perm(?:il)?p[sd]?|broadcast[is]?[dqbw]?)|
v?(?:dpps|rcpps|rsqrtpsd?)|
v?(?:pack[us]?[dw]|unpack[lh][psd])|
vmov[au]p[sd]|vmovdqa[32]?|vmovdqu[8|16|32]?|
v?p(?:add|sub|mul|min|max)[bwdq]|vpmadd(?:wd|ubsw)|vpshufb|
v?(?:gather|scatter)[dq][psd]?|
v?cvtt?(?:ps2dq|dq2ps|pd2dq|dq2pd)
""".replace("\n","")

SIMD_OP_RE = re.compile(rf"\b(?:{SIMD_OPS})\b", re.IGNORECASE)
REG_RE     = re.compile(r"\b[xyz]mm\d+\b")           # xmm0/ymm15/zmm31
WIDTH_RE   = re.compile(r"\b([xyz])mm(\d+)\b")       # capture vector class + index

def _extract_target(asm_dict, target=None):
    if isinstance(asm_dict, str):
        return asm_dict, "<string>"
    if not isinstance(asm_dict, dict):
        raise TypeError("inspect_asm() returned unexpected type")
    keys = list(asm_dict.keys())
    if not keys:
        raise ValueError("empty assembly dict")
    if target is None:
        target = keys[0]
    if target not in asm_dict:
        raise KeyError(f"target {target!r} not in {keys}")
    return asm_dict[target], target

def summarize_simd(asm_text: str):
    ops = Counter(
        (m[0].lower() if isinstance(m, tuple) else m.lower())
        for m in SIMD_OP_RE.findall(asm_text)
    )
    regs = REG_RE.findall(asm_text)
    # Detect widths by register class: xmm=128, ymm=256, zmm=512
    width_map = {"x":128, "y":256, "z":512}
    widths = Counter(width_map[m.group(1)] for m in WIDTH_RE.finditer(asm_text))
    return {
        "ops_total": sum(ops.values()),
        "ops_top": ops.most_common(15),
        "widths": dict(widths),            # e.g., {256: 120, 512: 40}
        "used_regs_count":   len(set(regs)), # unique regs touched
    }

def highlight_simd(asm_text: str) -> str:
    def color_ops(m):
        return f"{C['op']}{m.group(0)}{C['reset']}"
    def color_regs(m):
        return f"{C['reg']}{m.group(0)}{C['reset']}"
    out = SIMD_OP_RE.sub(color_ops, asm_text)
    out = REG_RE.sub(color_regs, out)
    return out

def show_simd_asm(asm_dict, target=None, show_summary=True, head=None):
    asm_text, tgt = _extract_target(asm_dict, target)
    if head is not None:
        asm_text = "\n".join(asm_text.splitlines()[:head])
    if show_summary:
        summary = summarize_simd(asm_text)
        badge = " ".join(
            f"{C['width']}{w}-bit×{n}{C['reset']}" for w,n in sorted(summary["widths"].items())
        ) or "no SIMD widths detected"
        print(f"{C['title']}[Target]{C['reset']} {tgt}  {badge}")
        print(f"{C['title']}[SIMD ops]{C['reset']} total={summary['ops_total']}, "
              f"unique regs={summary['used_regs_count']}")
        for op,count in summary["ops_top"]:
            print(f"  {op:24s}  {count}")
        print("-" * 60)
    print(highlight_simd(asm_text))
