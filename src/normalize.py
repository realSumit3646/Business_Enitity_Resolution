"""Text normalisation for business names and addresses.

Country-agnostic core (unicode folding, transliteration, leetspeak repair, junk removal) plus
abbreviation maps that contract every variant to one canonical short token, so that e.g.
"Street"/"St"/"STR" -> "st" and "Rue"/"R." -> "rue". French "Saint" also contracts to "st",
which is harmless because the mapping is consistent on both sides of a comparison.
"""
import json
import math
import re
import unicodedata
from pathlib import Path

from unidecode import unidecode

# ------------------------------------------------------------------ canonical maps
ADDR_MAP = {
    # street types (US / generic)
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "drv": "dr",
    "lane": "ln", "court": "ct", "crt": "ct", "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "bld": "blvd",
    "highway": "hwy", "hiway": "hwy", "parkway": "pkwy", "pky": "pkwy", "place": "pl", "circle": "cir",
    "trail": "trl", "terrace": "ter", "terr": "ter", "square": "sq", "suite": "ste", "apartment": "apt",
    "building": "bldg", "bldng": "bldg", "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "mount": "mt",
    "fort": "ft", "route": "rte", "point": "pt", "heights": "hts", "center": "ctr", "centre": "ctr",
    "expressway": "expy", "freeway": "fwy", "crossing": "xing", "junction": "jct", "plaza": "plz",
    "creek": "crk", "valley": "vly", "view": "vw", "village": "vlg", "ridge": "rdg", "hills": "hls",
    "hill": "hl", "lake": "lk", "park": "pk", "springs": "spgs", "station": "sta", "unit": "unit",
    "room": "rm", "department": "dept", "number": "no", "num": "no", "ndeg": "no", "numero": "no",
    # India
    "near": "nr", "opposite": "opp", "sector": "sec", "house": "h", "hno": "h no", "colony": "col",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "ground": "grd", "cross": "crs",
    "main": "mn", "nagar": "ngr", "marg": "mrg", "gali": "gl", "mohalla": "mhl", "village": "vlg",
    "post": "po", "tehsil": "teh", "district": "dist", "distt": "dist", "industrial": "ind", "estate": "est",
    "complex": "cplx", "apartments": "apt", "apts": "apt", "society": "soc", "chowk": "chk", "bazar": "bzr",
    "bazaar": "bzr", "market": "mkt",
    # France
    "r": "rue", "chemin": "chem", "ch": "chem", "impasse": "imp", "allee": "all", "quai": "qu",
    "cours": "crs", "saint": "st", "sainte": "ste", "faubourg": "fbg", "residence": "res", "batiment": "bat",
    "bat": "bat", "lieu": "lieu", "dit": "dit", "zone": "z",
}
NAME_MAP = {
    "private": "pvt", "pvt": "pvt", "prv": "pvt", "limited": "ltd", "ltd": "ltd", "ltd.": "ltd",
    "corporation": "corp", "incorporated": "inc", "company": "co", "cos": "co", "&": "and", "n": "and",
    "brothers": "bros", "international": "intl", "enterprises": "ent", "enterprise": "ent",
    "association": "assn", "assoc": "assn", "services": "svc", "service": "svc", "svcs": "svc",
    "technologies": "tech", "technology": "tech", "industries": "ind", "manufacturing": "mfg",
    "management": "mgmt", "solutions": "sol", "solution": "sol", "center": "ctr", "centre": "ctr",
    "saint": "st", "sainte": "ste", "et": "and", "und": "and", "y": "and", "compagnie": "cie",
}
LEGAL = {
    "pvt", "ltd", "corp", "co", "inc", "llc", "llp", "lp", "pc", "pllc", "plc", "pa", "sa", "sas", "sasu",
    "sarl", "eurl", "sci", "snc", "scp", "selarl", "selas", "scm", "gie", "ei", "cie", "gmbh", "opc", "dba", "lc", "ltda", "the", "and",
    "of", "de", "du", "des", "la", "le", "les", "l", "d",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "dl": "delhi",
}
# Full state names -> postal codes, applied as phrases on the folded address. Codes collide across countries
# (e.g. "tn" Tennessee / Tamil Nadu) which is harmless because matching never crosses countries.
STATE_PHRASES = {**US_STATES, **{k: v for k, v in IN_STATES.items() if " " in k or len(k) > 3}}
STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_PHRASES), key=len, reverse=True)) + r")\b")

# Admin-area rewrites learned from unlabelled data (learn_admin_map.py), e.g. French department -> region.
try:
    _ADMIN = json.load(open(Path(__file__).resolve().parent / "resources" / "admin_map.json", encoding="utf-8"))
except FileNotFoundError:
    _ADMIN = {}
ADMIN_RE = {c: (re.compile(r"\b(" + "|".join(sorted(map(re.escape, m), key=len, reverse=True)) + r")\b"), m)
            for c, m in _ADMIN.items() if m}

_TMAP_PATH = Path(__file__).resolve().parent / "resources" / "translit_map.json"
try:
    _T = json.load(open(_TMAP_PATH, encoding="utf-8"))
except FileNotFoundError:  # before learn_translit.py has been run
    _T = {"name": {}, "addr": {}}
TRANSLIT_NAME = {**_T["addr"], **_T["name"]}
TRANSLIT_ADDR = {**_T["name"], **_T["addr"]}

# S1 name vocabulary (build_vocab.py) for splitting concatenated names; canonical forms also get their
# original spellings ("pvt" -> "private") so "ssestateprivate" splits into "ss estate private".
try:
    VOCAB = json.load(open(Path(__file__).resolve().parent / "resources" / "name_vocab.json"))
except FileNotFoundError:
    VOCAB = {}
for _orig, _canon in list(NAME_MAP.items()):
    if _canon in VOCAB and _orig.isalpha() and len(_orig) >= 2:
        VOCAB[_orig] = max(VOCAB.get(_orig, 0), VOCAB[_canon])
_VTOTAL = math.log(sum(VOCAB.values()) or 1)
UNK_COST = 10.0  # per character; any in-vocabulary word is far cheaper


def _wcost(w):
    c = VOCAB.get(w)
    return _VTOTAL - math.log(c) if c else None


def segment(s: str, max_len: int = 20):
    """Viterbi split of a concatenated lowercase string into S1-vocabulary words.
    Returns (tokens, n_unknown_chars); unknown characters are grouped into one token per run."""
    n = len(s)
    best = [0.0] + [math.inf] * n
    back = [(0, False)] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            c = _wcost(s[j:i])
            if c is not None and best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, (j, True)
        if best[i - 1] + UNK_COST < best[i]:
            best[i], back[i] = best[i - 1] + UNK_COST, (i - 1, False)
    toks, unk, i, run = [], 0, n, ""
    while i > 0:
        j, known = back[i]
        if known:
            if run:
                toks.append(run)
                run = ""
            toks.append(s[j:i])
        else:
            run = s[j:i] + run
            unk += 1
        i = j
    if run:
        toks.append(run)
    return toks[::-1], unk


def split_concatenated(toks, is_domain):
    """Split a website name entirely; split other long out-of-vocabulary words only when they split cleanly."""
    if not VOCAB:
        return toks
    if is_domain:
        seg, unk = segment("".join(toks))
        return seg if unk <= max(2, len("".join(toks)) // 4) else toks
    out = []
    for t in toks:
        if len(t) >= 10 and t.isalpha() and t not in VOCAB:
            seg, unk = segment(t)
            if len(seg) >= 2 and unk <= 2:
                out.extend(seg)
                continue
        out.append(t)
    return out


# Words the synthetic generator ("Dr", "Services", "Group", ...) are left in;
# the model learns their (low) weight through IDF-based features.

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
URL_TAIL = re.compile(r"\s*[|\-–]\s*(https?://|www\.)\S*.*$", re.I)
URL_ANY = re.compile(r"(https?://)?(www\.)?", re.I)
DOMAIN = re.compile(r"^\s*(?:https?://)?(?:www\.)?([\w\-]+)\.(?:com|in|net|org|co|fr|biz|info|us|co\.in|io)\s*$", re.I)
NA = re.compile(r"\b(n\s*/\s*a|null|none)\b", re.I)
DOTTED = re.compile(r"\b[a-z](?:\.[a-z])+\.?(?!\w)")   # "l.l.c." -> "llc", "s.a.s.u." -> "sasu"
POBOX = re.compile(r"\b(p\.?\s*o\.?\s*box|pmb|post\s*box)\s*#?\s*\d+\b", re.I)
NONWORD = re.compile(r"[^\w]+")
ALNUM_SPLIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")   # "b1" -> "b 1", "19d" -> "19 d"
VOWELS = re.compile(r"[aeiouy]")
DOUBLE = re.compile(r"(.)\1+")
SKEL_SUBS = [("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "j"), ("x", "ks")]


def fold(s: str) -> str:
    """Unicode -> lowercase ASCII (accents stripped, Indic scripts romanised)."""
    s = unicodedata.normalize("NFKC", s).replace("‌", "").replace("‍", "")
    return unidecode(s).lower()


def has_nonlatin(s: str) -> bool:
    return any(ord(c) > 0x24F and c.isalpha() for c in s)


def fold_mapped(s: str, tmap: dict) -> str:
    """fold(), but romanised non-Latin words are passed through the learned translit map."""
    if not has_nonlatin(s):
        return fold(s)
    out = []
    for w in s.split():
        f = fold(w)
        if has_nonlatin(w):
            f = " ".join(tmap.get(t, t) for t in NONWORD.sub(" ", f).split())
        out.append(f)
    return " ".join(out)


def strip_zeros(tok: str) -> str:
    return (tok.lstrip("0") or "0") if tok.isdigit() else tok


def fix_leet(tok: str, strict: bool) -> str:
    if not tok or tok.isdigit() or tok.isalpha() or ORDINAL.match(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = len(tok) - letters
    if strict and not (letters >= 3 and digits == 1):
        return tok
    if letters >= 2:
        return tok.translate(LEET)
    return tok


def dedup_consecutive(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


def skeleton(tok: str) -> str:
    """Consonant skeleton that is stable across English spelling and rough Indic romanisation."""
    for a, b in SKEL_SUBS:
        tok = tok.replace(a, b)
    tok = tok.replace("h", "") if len(tok) > 2 else tok
    tok = DOUBLE.sub(r"\1", tok)
    head, tail = tok[:1], VOWELS.sub("", tok[1:])
    return head + tail


_CANON_WORDS = {**{k: v for k, v in NAME_MAP.items() if len(k) > 3}, "llc": "llc", "limited": "ltd"}
# romanised Indic forms of canonical words, matched on consonant skeleton (e.g. "praaivett" -> "prvt")
SKEL_CANON = {skeleton(k): v for k, v in _CANON_WORDS.items()} | {"ellp": "llp", "elelp": "llp", "ltdd": "ltd"}


def norm_name(raw: str):
    """Return dict with name variants."""
    raw = raw or ""
    nonlatin = has_nonlatin(raw)
    s = URL_TAIL.sub("", raw)
    m = DOMAIN.match(s)
    is_domain = bool(m)
    if m:
        s = m.group(1)
    s = fold_mapped(s, TRANSLIT_NAME)
    s = URL_ANY.sub("", s)
    s = DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("'", "")
    toks = [fix_leet(t, strict=False) for t in NONWORD.sub(" ", s).split()]
    if nonlatin:
        toks = [SKEL_CANON.get(skeleton(t), t) for t in toks]
    toks = split_concatenated(toks, is_domain)
    toks = dedup_consecutive([NAME_MAP.get(t, t) for t in toks])
    legal = sorted({t for t in toks if t in LEGAL and t not in {"the", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d"}})
    core = [t for t in toks if t not in LEGAL] or toks
    return {
        "name": " ".join(toks),
        "name_core": " ".join(core),
        "name_compact": "".join(core),
        "name_skel": " ".join(skeleton(t) for t in core),
        "legal": " ".join(legal),
        "is_domain": is_domain,
        "name_nonlatin": nonlatin,
    }


def norm_addr(raw: str, country: str = ""):
    raw = raw or ""
    nonlatin = has_nonlatin(raw)
    s = NA.sub(" ", raw)
    pobox = bool(POBOX.search(s))
    s = POBOX.sub(" ", s)
    s = fold_mapped(s, TRANSLIT_ADDR).replace("&", " and ").replace("'", " ")
    s = NONWORD.sub(" ", s)
    if country in ADMIN_RE:
        rx, m_ = ADMIN_RE[country]
        s = rx.sub(lambda m: m_[m.group(1)], s)
    s = STATE_RE.sub(lambda m: STATE_PHRASES[m.group(1)], s)
    toks = [strip_zeros(fix_leet(t, strict=True)) for t in s.split()]
    toks = " ".join(ALNUM_SPLIT.sub(" ", t) if not ORDINAL.match(t) else t for t in toks).split()
    toks = [strip_zeros(t) for t in toks]
    toks = [IN_STATES.get(t, t) if len(t) == 2 else t for t in toks]
    toks = " ".join(ADDR_MAP.get(t, t) for t in toks).split()
    toks = dedup_consecutive(toks)
    nums = sorted({t for t in toks if t.isdigit()})
    words = [t for t in toks if not t.isdigit()]
    return {
        "addr": " ".join(toks),
        "addr_words": " ".join(words),
        "addr_nums": " ".join(nums),
        "addr_skel": " ".join(skeleton(t) for t in words if len(t) > 1),
        "addr_empty": len(toks) == 0,
        "addr_pobox": pobox,
        "addr_nonlatin": nonlatin,
    }


def normalize_record(name: str, addr: str, country: str = ""):
    d = norm_name(name)
    d.update(norm_addr(addr, country))
    return d


if __name__ == "__main__":
    tests = [
        ("*** THE BROWN & WARFIELD LLC", "N/A, CT, GRISWOLD, PLAINFIELD RD"),
        ("kimb1eolvas.com", "607 VIRGINIA ST, PO BOX 2342, TERRELL, TX"),
        ("Cardiology 5afe Care Inc", "617 FIREHOUSE ROAD, FLOYD COUNTY, VA"),
        ("अल टेक प्राइवेट लिमिटेड", "1605/ 1606, Vashi, MH, Thane"),
        ("Al Tech Private Limited", "The Ambidence Court Plot No. 2, Sector 19D, Navi Mumbai"),
        ("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com", "H.NO 204 C ROAD HOSHIARPUR"),
        ("Rotary Çentre SARL", "061 RUE MAURIAN, Nouvelle-Aquitaine, MÉRIGNAC"),
        ("établissements Ass Sas", "Roubaix, (27) R. Villars"),
        ("Secunderabad Easy Pdrlvate Limited", "Flat 203, 12Th Main, 7Th Cross, B T M 2Nd Stage, ಕರ್ನಾಟಕ"),
    ]
    for n, a in tests:
        print(n, "|", a)
        for k, v in normalize_record(n, a).items():
            print(f"    {k:14s} {v!r}")
