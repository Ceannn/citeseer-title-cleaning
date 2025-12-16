import math
import re
from typing import Dict, List

import numpy as np

NOISE_KW = [
    "call for papers",
    "fragments",
    "proceedings",
    "volume",
    "issue",
    "pages",
    "copyright",
]

_token_re = re.compile(r"\w+")
_punct_re = re.compile(r"[^\w\s]")


def _max_run(s: str) -> int:
    """Return max run length of the same char."""
    if not s:
        return 0
    max_len = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 1
    return max_len


def extract_features(title: str) -> np.ndarray:
    """Generate cheap features vector of length 8."""
    text = str(title)
    n = len(text)
    tokens = _token_re.findall(text)
    n_tokens = len(tokens)
    digit_count = sum(c.isdigit() for c in text)
    punct_count = len(_punct_re.findall(text))
    upper_count = sum(c.isupper() for c in text)
    space_count = sum(c.isspace() for c in text)

    len_chars = math.log1p(n)
    len_tokens = n_tokens
    digit_ratio = digit_count / n if n > 0 else 0.0
    punct_ratio = punct_count / n if n > 0 else 0.0
    upper_ratio = upper_count / n if n > 0 else 0.0
    space_ratio = space_count / n if n > 0 else 0.0
    repeat_char_maxrun = _max_run(text)
    contains_noise_kw = 1 if any(kw in text.lower() for kw in NOISE_KW) else 0

    return np.array(
        [
            len_chars,
            len_tokens,
            digit_ratio,
            punct_ratio,
            upper_ratio,
            space_ratio,
            repeat_char_maxrun,
            contains_noise_kw,
        ],
        dtype=np.float32,
    )


def batch_extract(titles: List[str]) -> np.ndarray:
    feats = [extract_features(t) for t in titles]
    return np.vstack(feats)


def feature_names() -> List[str]:
    return [
        "len_chars_log1p",
        "len_tokens",
        "digit_ratio",
        "punct_ratio",
        "upper_ratio",
        "space_ratio",
        "repeat_char_maxrun",
        "contains_noise_kw",
    ]


def feature_dict(title: str) -> Dict[str, float]:
    arr = extract_features(title)
    return dict(zip(feature_names(), arr.tolist()))
