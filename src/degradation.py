"""
Step 2 — Degradation Engine  (patched)
Corrupts clean texts along four independent, orthogonal axes at controlled
intensity levels.

Axes
────
1. Grammar   — keyboard typos, agreement errors, tense swaps, article misuse,
               homophones, preposition errors, bad comparatives,
               hyphenation errors, word-order disruption
2. Coherence — sentence-order shuffling (preserves all content)
3. Information — word/phrase/clause deletion (modifiers → parentheticals
                 → subordinate clauses → content words)
4. Lexical   — vocabulary flattening via WordNet synonym collapse

Patches applied vs original
────────────────────────────
FIX 1 – Reproducibility: replaced Python's `hash()` (non-deterministic across
         sessions due to PYTHONHASHSEED) with hashlib.md5 so seeds are
         identical every run, on every machine, forever.

FIX 2 – Detokenization: replaced the hand-rolled punctuation-gluing loop in
         degrade_lexical Phase 3 with NLTK's TreebankWordDetokenizer, which
         perfectly inverts word_tokenize and avoids spurious spaces that would
         bleed Axis 4 artifacts into Axis 1 signal.
"""

import hashlib
import json
import random
import re
from pathlib import Path

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

# ── NLTK resource bootstrap ────────────────────────────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers", "punkt"),
    ("tokenizers", "punkt_tab"),
    ("taggers",    "averaged_perceptron_tagger"),
    ("taggers",    "averaged_perceptron_tagger_eng"),
    ("corpora",    "wordnet"),
]
for folder, name in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"{folder}/{name}")
    except LookupError:
        nltk.download(name, quiet=True)

_DETOKENIZER = TreebankWordDetokenizer()


# ═══════════════════════════════════════════════════════════════════
# Axis 1: Grammar Corruption
# ═══════════════════════════════════════════════════════════════════

_CONFUSABLES = {
    "its": "it's",  "it's": "its",
    "their": "there", "there": "their",
    "they're": "their", "your": "you're", "you're": "your",
    "affect": "effect", "effect": "affect",
    "then": "than", "than": "then",
    "who's": "whose", "whose": "who's",
    "to": "too", "too": "to",
    "accept": "except", "except": "accept",
    "lose": "loose", "loose": "lose",
    "complement": "compliment", "compliment": "complement",
    "principal": "principle", "principle": "principal",
    "stationary": "stationery", "stationery": "stationary",
    "weather": "whether", "whether": "weather",
    "cite": "site", "site": "cite",
}

_WRONG_PREPS = {
    "interested in":  "interested for",
    "depend on":      "depend of",
    "capable of":     "capable to",
    "consist of":     "consist from",
    "according to":   "according with",
    "result in":      "result to",
    "different from": "different than",
    "similar to":     "similar with",
    "aware of":       "aware about",
    "based on":       "based off",
    "responsible for":"responsible of",
    "refer to":       "refer at",
    "related to":     "related with",
    "arrived at":     "arrived to",
    "independent of": "independent from",
}

_AGREEMENT_SUBS = [
    (r'\b(he|she|it) (was)\b',  r'\1 were'),
    (r'\b(they) (were)\b',      r'\1 was'),
    (r'\b(he|she|it) (has)\b',  r'\1 have'),
    (r'\b(they) (have)\b',      r'\1 has'),
    (r'\b(he|she|it) (does)\b', r'\1 do'),
    (r'\b(they) (do)\b',        r'\1 does'),
    (r'\b(he|she|it) (is)\b',   r'\1 are'),
    (r'\b(they) (are)\b',       r'\1 is'),
]

_TENSE_SWAPS = [
    (r'\b(was)\b',    'is'),    (r'\b(were)\b',    'are'),
    (r'\b(had)\b',    'has'),   (r'\b(did)\b',     'does'),
    (r'\b(went)\b',   'goes'),  (r'\b(came)\b',    'comes'),
    (r'\b(took)\b',   'takes'), (r'\b(made)\b',    'makes'),
    (r'\b(said)\b',   'says'),  (r'\b(became)\b',  'becomes'),
    (r'\b(began)\b',  'begins'),(r'\b(gave)\b',    'gives'),
    (r'\b(found)\b',  'finds'), (r'\b(built)\b',   'builds'),
    (r'\b(led)\b',    'leads'), (r'\b(wrote)\b',   'writes'),
    (r'\b(ran)\b',    'runs'),  (r'\b(held)\b',    'holds'),
    (r'\b(brought)\b','brings'),(r'\b(fought)\b',  'fights'),
]

_BAD_COMPARATIVES = {
    "better": "more better", "worse":  "more worse",
    "bigger": "more bigger", "smaller":"more smaller",
    "faster": "more faster", "larger": "more larger",
    "easier": "more easier", "harder": "more harder",
}

_HYPHEN_SPLITS = [
    "well-known","long-term","high-quality","so-called","self-made",
    "non-profit","full-time","part-time","large-scale","small-scale",
    "well-established","short-lived","long-standing","wide-ranging",
    "far-reaching","hard-working","old-fashioned","open-ended",
    "state-of-the-art","up-to-date","first-hand","second-hand",
    "man-made","hand-made","re-elected","co-authored",
]
_HYPHEN_JOINS = [
    ("every day","everyday"),("any time","anytime"),
    ("every one","everyone"),("some times","sometimes"),
    ("in to","into"),("on to","onto"),
]


def _apply_keyboard_typos(text: str, level: float, rng: random.Random) -> str:
    char_p = min(level * 0.02, 0.03)
    adjacent_keys = {
        'a':'sq','b':'vn','c':'xv','d':'sf','e':'wr','f':'dg','g':'fh',
        'h':'gj','i':'uo','j':'hk','k':'jl','l':'k','m':'n','n':'bm',
        'o':'ip','p':'o','q':'w','r':'et','s':'ad','t':'ry','u':'yi',
        'v':'cb','w':'qe','x':'zc','y':'tu','z':'x',
    }
    result = []
    for ch in text:
        if ch.isalpha() and rng.random() < char_p:
            action = rng.choice(["swap","double","drop"])
            lower  = ch.lower()
            if action == "swap" and lower in adjacent_keys:
                r = rng.choice(adjacent_keys[lower])
                result.append(r if ch.islower() else r.upper())
            elif action == "double":
                result.append(ch); result.append(ch)
            # drop: append nothing
        else:
            result.append(ch)
    return "".join(result)


def _apply_agreement_errors(text: str, level: float, rng: random.Random) -> str:
    for pattern, replacement in _AGREEMENT_SUBS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if rng.random() < level * 0.3:
                new  = re.sub(pattern, replacement, m.group(0), flags=re.IGNORECASE)
                text = text[:m.start()] + new + text[m.end():]
    return text


def _apply_tense_swaps(text: str, level: float, rng: random.Random) -> str:
    for pattern, replacement in _TENSE_SWAPS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if rng.random() < level * 0.15:
                new  = replacement.capitalize() if m.group(0)[0].isupper() else replacement
                text = text[:m.start()] + new + text[m.end():]
    return text


def _apply_article_errors(text: str, level: float, rng: random.Random) -> str:
    words  = text.split()
    result = []
    i      = 0
    while i < len(words):
        w     = words[i]
        lower = w.lower().rstrip(".,;:!?")
        if lower in ("a","an") and rng.random() < level * 0.3:
            new   = "an" if lower == "a" else "a"
            trail = w[len(lower):]
            result.append((new.capitalize() if w[0].isupper() else new) + trail)
        elif lower in ("the","a","an") and rng.random() < level * 0.15:
            pass  # drop article
        elif (w[0].isupper() and i > 0
              and not words[i-1].endswith(".")
              and rng.random() < level * 0.05):
            result.append("the"); result.append(w)
        else:
            result.append(w)
        i += 1
    return " ".join(result)


def _apply_confusables(text: str, level: float, rng: random.Random) -> str:
    words  = text.split()
    result = []
    for w in words:
        stripped = w.lower().rstrip(".,;:!?\"')")
        trail    = w[len(stripped):]
        if stripped in _CONFUSABLES and rng.random() < level * 0.25:
            r = _CONFUSABLES[stripped]
            result.append((r.capitalize() if w[0].isupper() else r) + trail)
        else:
            result.append(w)
    return " ".join(result)


def _apply_preposition_errors(text: str, level: float, rng: random.Random) -> str:
    for correct, wrong in _WRONG_PREPS.items():
        if correct in text.lower() and rng.random() < level * 0.4:
            idx = text.lower().find(correct)
            if idx >= 0:
                text = text[:idx] + wrong + text[idx + len(correct):]
    return text


def _apply_comparative_errors(text: str, level: float, rng: random.Random) -> str:
    for correct, wrong in _BAD_COMPARATIVES.items():
        for m in reversed(list(re.finditer(rf'\b{correct}\b', text, re.IGNORECASE))):
            if rng.random() < level * 0.5:
                text = text[:m.start()] + wrong + text[m.end():]
    return text


def _apply_hyphenation_errors(text: str, level: float, rng: random.Random) -> str:
    for compound in _HYPHEN_SPLITS:
        if compound in text.lower() and rng.random() < level * 0.4:
            idx = text.lower().find(compound)
            if idx >= 0:
                text = text[:idx] + text[idx:idx+len(compound)].replace("-"," ") + text[idx+len(compound):]
    for separate, joined in _HYPHEN_JOINS:
        if separate in text.lower() and rng.random() < level * 0.3:
            idx = text.lower().find(separate)
            if idx >= 0:
                text = text[:idx] + joined + text[idx+len(separate):]
    return text


def _apply_word_order_disruption(text: str, level: float, rng: random.Random) -> str:
    if level < 0.3:
        return text
    sentences = nltk.sent_tokenize(text)
    result    = []
    for sent in sentences:
        words   = sent.split()
        n_swaps = max(0, int(len(words) * level * 0.08))
        for _ in range(n_swaps):
            idx = rng.randint(0, max(0, len(words)-2))
            words[idx], words[idx+1] = words[idx+1], words[idx]
        result.append(" ".join(words))
    return " ".join(result)


def degrade_grammar(text: str, level: float, rng: random.Random) -> str:
    if level <= 0:
        return text
    text = _apply_keyboard_typos(text, level, rng)
    text = _apply_agreement_errors(text, level, rng)
    text = _apply_tense_swaps(text, level, rng)
    text = _apply_article_errors(text, level, rng)
    text = _apply_confusables(text, level, rng)
    text = _apply_preposition_errors(text, level, rng)
    text = _apply_comparative_errors(text, level, rng)
    text = _apply_hyphenation_errors(text, level, rng)
    text = _apply_word_order_disruption(text, level, rng)
    return text


# ═══════════════════════════════════════════════════════════════════
# Axis 2: Coherence — Sentence-Order Shuffling
# ═══════════════════════════════════════════════════════════════════

def degrade_coherence(text: str, level: float, rng: random.Random) -> str:
    """Shuffle sentence order to break logical flow and coreference chains.
    Grammar, vocabulary, and information are fully preserved."""
    if level <= 0:
        return text
    sentences = nltk.sent_tokenize(text)
    n         = len(sentences)
    if n <= 2:
        return text

    n_swaps          = max(1, int(n * level * 0.8))
    max_displacement = max(1, int(n * level * 0.6))
    indices          = list(range(n))

    for _ in range(n_swaps):
        i     = rng.randint(0, n-1)
        j_min = max(0, i - max_displacement)
        j_max = min(n-1, i + max_displacement)
        j     = rng.randint(j_min, j_max)
        indices[i], indices[j] = indices[j], indices[i]

    return " ".join(sentences[i] for i in indices)


# ═══════════════════════════════════════════════════════════════════
# Axis 3: Information — Word/Phrase/Clause Deletion
# ═══════════════════════════════════════════════════════════════════

_MODIFIER_TAGS = {"JJ","JJR","JJS","RB","RBR","RBS"}
_CONTENT_TAGS  = {
    "NN","NNS","NNP","NNPS",
    "VB","VBD","VBG","VBN","VBP","VBZ",
    "JJ","JJR","JJS","RB","RBR","RBS",
}

_PARENTHETICAL_RE  = re.compile(r'\([^)]{5,80}\)')
_SUBORD_CLAUSE_RE  = re.compile(
    r'\b(?:which|who|whom|that|although|though|because|since|while|whereas|'
    r'unless|whenever|wherever|if|when|after|before|until)\b[^.;]{10,80}[,.]',
    re.IGNORECASE,
)
_PREP_PHRASE_RE = re.compile(
    r'\b(?:in|on|at|by|for|with|from|about|during|through|between|among|'
    r'under|above|after|before|without|within|across|behind|beyond)\s+'
    r'(?:the\s+|a\s+|an\s+)?(?:\w+\s*){1,5}',
    re.IGNORECASE,
)


def _delete_modifiers(text: str, level: float, rng: random.Random) -> str:
    sentences = nltk.sent_tokenize(text)
    result    = []
    for sent in sentences:
        try:
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        except Exception:
            result.append(sent); continue
        kept = [w for w, tag in tagged
                if not (tag in _MODIFIER_TAGS and rng.random() < level * 0.5)]
        result.append(" ".join(kept))
    return " ".join(result)


def _delete_parentheticals(text: str, level: float, rng: random.Random) -> str:
    return _PARENTHETICAL_RE.sub(
        lambda m: "" if rng.random() < level * 0.6 else m.group(0), text
    )


def _delete_subordinate_clauses(text: str, level: float, rng: random.Random) -> str:
    def maybe_remove(m):
        if rng.random() < level * 0.3:
            ending = m.group(0)[-1]
            return ending if ending in ".," else ""
        return m.group(0)
    return _SUBORD_CLAUSE_RE.sub(maybe_remove, text)


def _delete_content_words(text: str, level: float, rng: random.Random) -> str:
    if level < 0.5:
        return text
    sentences = nltk.sent_tokenize(text)
    result    = []
    for sent in sentences:
        try:
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        except Exception:
            result.append(sent); continue
        kept = [w for w, tag in tagged
                if not (tag in _CONTENT_TAGS and tag not in _MODIFIER_TAGS
                        and rng.random() < (level - 0.4) * 0.2)]
        if kept:
            result.append(" ".join(kept))
    return " ".join(result)


def _delete_prep_phrases(text: str, level: float, rng: random.Random) -> str:
    if level < 0.3:
        return text
    return _PREP_PHRASE_RE.sub(
        lambda m: "" if rng.random() < (level - 0.2) * 0.25 else m.group(0), text
    )


def degrade_information(text: str, level: float, rng: random.Random) -> str:
    if level <= 0:
        return text
    text = _delete_parentheticals(text, level, rng)
    text = _delete_modifiers(text, level, rng)
    text = _delete_prep_phrases(text, level, rng)
    text = _delete_subordinate_clauses(text, level, rng)
    text = _delete_content_words(text, level, rng)
    # Clean up artifacts
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Axis 4: Lexical Resource Collapse
# ═══════════════════════════════════════════════════════════════════

_WORDNET_POS_MAP = {
    "NN":"n","NNS":"n",
    "VB":"v","VBD":"v","VBG":"v","VBN":"v","VBP":"v","VBZ":"v",
    "JJ":"a","JJR":"a","JJS":"a",
    "RB":"r","RBR":"r","RBS":"r",
}

_WORDNET_SKIP = {
    "is","are","was","were","be","been","being","am","has","have","had",
    "do","does","did","will","would","shall","should","can","could","may",
    "might","must","the","a","an","and","or","but","in","on","at","to",
    "for","of","with","by","from","not","no","it","its","he","she","they",
    "we","you","i","me","him","her","them","us","my","his","our","your",
    "their","this","that","these","those","which","who","whom","whose",
    "what","where","when","how","why","if","then","than","as","so","very",
    "also","just","only","more","most","much","many","some","any","all",
    "each","every","both","few","other","such","own","same","first","last",
    "new","old","good","great","little","big","long","after","before",
    "between","over","under","about","up","down","out","into","through",
    "during","without","within","short","hard","fast","well","right",
    "left","light","close","late","early","still","even","back","round",
    "like","near","open","free","fine","clear","full","high","low","true",
    "real","sure","dead","live",
}

_IRREGULAR_PAST = {
    "get":"got","say":"said","make":"made","know":"knew","come":"came",
    "take":"took","give":"gave","find":"found","tell":"told","show":"showed",
    "go":"went","see":"saw","run":"ran","build":"built","hold":"held",
    "write":"wrote","keep":"kept","lead":"led","meet":"met","pay":"paid",
    "leave":"left","bring":"brought","begin":"began","grow":"grew",
    "draw":"drew","break":"broke","speak":"spoke","drive":"drove",
    "rise":"rose","choose":"chose","fall":"fell","bear":"bore",
    "think":"thought","feel":"felt","send":"sent","stand":"stood",
    "lose":"lost","cut":"cut","put":"put","set":"set","let":"let",
    "do":"did","have":"had","be":"was","control":"controlled",
    "admit":"admitted","occur":"occurred","refer":"referred",
    "permit":"permitted","submit":"submitted","prefer":"preferred",
    "commit":"committed","omit":"omitted",
}
_IRREGULAR_PARTICIPLE = {
    "get":"gotten","say":"said","make":"made","know":"known","come":"come",
    "take":"taken","give":"given","find":"found","tell":"told","show":"shown",
    "go":"gone","see":"seen","run":"run","build":"built","hold":"held",
    "write":"written","keep":"kept","lead":"led","meet":"met","pay":"paid",
    "leave":"left","bring":"brought","begin":"begun","grow":"grown",
    "draw":"drawn","break":"broken","speak":"spoken","drive":"driven",
    "rise":"risen","choose":"chosen","fall":"fallen","bear":"borne",
    "think":"thought","feel":"felt","send":"sent","stand":"stood",
    "lose":"lost","cut":"cut","put":"put","set":"set","let":"let",
    "do":"done","have":"had","be":"been","control":"controlled",
    "admit":"admitted","occur":"occurred","refer":"referred",
    "permit":"permitted","submit":"submitted","prefer":"preferred",
    "commit":"committed","omit":"omitted",
}


def _needs_doubling(base: str) -> bool:
    if len(base) < 3:
        return False
    vowels  = set("aeiou")
    groups  = 0
    in_vowel = False
    for c in base:
        if c in vowels:
            if not in_vowel:
                groups += 1; in_vowel = True
        else:
            in_vowel = False
    return (groups == 1 and base[-1] not in vowels
            and base[-1] not in "wxy"
            and base[-2] in vowels and base[-3] not in vowels)


def _transfer_morphology(original: str, base_replacement: str, tag: str) -> str:
    r = base_replacement
    rl = r.lower()
    if tag in ("NN","JJ","JJR","JJS","RB","RBR","RBS"):
        return r
    if tag == "NNS":
        return r if r.endswith("s") else r + "s"
    if tag == "VBD":
        if rl in _IRREGULAR_PAST: return _IRREGULAR_PAST[rl]
        if rl.endswith("e"):      return r + "d"
        if _needs_doubling(rl):   return r + r[-1] + "ed"
        return r + "ed"
    if tag == "VBN":
        if rl in _IRREGULAR_PARTICIPLE: return _IRREGULAR_PARTICIPLE[rl]
        if rl.endswith("e"):            return r + "d"
        if _needs_doubling(rl):         return r + r[-1] + "ed"
        return r + "ed"
    if tag == "VBG":
        if rl.endswith("e"):    return r[:-1] + "ing"
        if _needs_doubling(rl): return r + r[-1] + "ing"
        return r + "ing"
    if tag == "VBZ":
        return r + "es" if rl.endswith(("s","x","z","ch","sh")) else r + "s"
    return r


def _get_lemma(word: str, tag: str) -> str:
    from nltk.stem import WordNetLemmatizer
    wn_pos = _WORDNET_POS_MAP.get(tag, "n")
    return WordNetLemmatizer().lemmatize(word.lower(), pos=wn_pos)


# ── FastText + wordfreq synonym engine ────────────────────────────────────────
# Uses pre-trained FastText/GloVe word vectors for synonym candidate generation.
# wordfreq filters out technical/rare terms before any substitution.
# Same-POS constraint + cosine similarity threshold ensure quality.
#
# Required: download ONE of these vector files and place next to this script:
#   GloVe 840B 300d (~2.2GB):
#     https://nlp.stanford.edu/data/glove.840B.300d.zip
#   FastText English (~2.2GB):
#     https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
#
# Then set VECTOR_FILE below to the filename.

VECTOR_FILE = "glove.840B.300d.txt"   # change to your downloaded file

# Similarity threshold — pairs below this are not synonyms
COSINE_THRESHOLD = 0.65

# wordfreq threshold — words below this are technical/rare, skip them
# ~3e-6 covers the top ~100k most common English words
FREQ_THRESHOLD = 3e-6

# Number of nearest neighbors to retrieve per word
TOP_N = 30

_VECTORS = None   # gensim KeyedVectors, loaded once on first use


def _load_vectors() -> None:
    """Load word vectors into _VECTORS. Called once on first use."""
    global _VECTORS
    if _VECTORS is not None:
        return

    import os
    from gensim.models import KeyedVectors

    candidates = [VECTOR_FILE,
                  os.path.join(os.path.dirname(__file__), VECTOR_FILE)]
    chosen = next((p for p in candidates if os.path.exists(p)), None)

    if chosen is None:
        raise FileNotFoundError(
            f"{VECTOR_FILE} not found. Download GloVe 840B 300d from: "
            "https://nlp.stanford.edu/data/glove.840B.300d.zip "
            "or FastText from: "
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz "
            "and place the extracted file next to degradation_engine.py, "
            "then set VECTOR_FILE at the top of this section."
        )

    # Prefer the fast .kv format (produced by convert_glove.py)
    kv_path = chosen.replace(".txt", ".kv").replace(".bin", ".kv")
    if os.path.exists(kv_path):
        print(f"[Lexical] Loading from fast cache {kv_path} ...")
        _VECTORS = KeyedVectors.load(kv_path, mmap="r")
    elif chosen.endswith(".bin"):
        print(f"[Lexical] Loading binary vectors from {chosen} ...")
        _VECTORS = KeyedVectors.load_word2vec_format(chosen, binary=True)
    else:
        print(f"[Lexical] Loading {chosen} (slow — run convert_glove.py first) ...")
        _VECTORS = KeyedVectors.load_word2vec_format(chosen, binary=False,
                                                      no_header=True)
        # Auto-save .kv so next run is fast
        print(f"[Lexical] Saving fast cache to {kv_path} ...")
        _VECTORS.save(kv_path)
        print(f"[Lexical] Saved. Future runs will load in ~30 seconds.")
    print(f"[Lexical] Loaded {len(_VECTORS):,} vectors.")


# POS tag groups — only substitute within same broad POS class
_POS_GROUP = {
    "NN": "noun",  "NNS": "noun",
    "VB": "verb",  "VBD": "verb",  "VBG": "verb",
    "VBN": "verb", "VBP": "verb",  "VBZ": "verb",
    "JJ": "adj",   "JJR": "adj",   "JJS": "adj",
}


# Cache for synonym lookups — each (word, pos_group) pair is looked up once
# across all articles, axes, levels and reps, cutting runtime dramatically.
_SYNONYM_CACHE: dict[tuple[str, str], set[str]] = {}
_TECHNICAL_CACHE: dict[str, bool] = {}


def _is_technical(word: str) -> bool:
    """Return True if word is too rare/technical for substitution."""
    key = word.lower()
    if key not in _TECHNICAL_CACHE:
        from wordfreq import word_frequency
        _TECHNICAL_CACHE[key] = word_frequency(key, "en") < FREQ_THRESHOLD
    return _TECHNICAL_CACHE[key]


def _get_embedding_synonyms(word: str, pos_tag: str) -> set[str]:
    """
    Get synonym candidates for `word` using word vectors.
    Results are cached by (word, pos_group) so each unique word is looked
    up only once across all 9,000 samples.

    Filters applied to each neighbor:
      1. Same broad POS group (noun/verb/adj) via the source word's tag
      2. wordfreq frequency >= FREQ_THRESHOLD (not a technical/rare term)
      3. Cosine similarity >= COSINE_THRESHOLD
      4. Single alpha token, length >= 4, not in skip list
    """
    _load_vectors()
    key = word.lower()
    pos_group = _POS_GROUP.get(pos_tag)
    if pos_group is None:
        return set()

    cache_key = (key, pos_group)
    if cache_key in _SYNONYM_CACHE:
        return _SYNONYM_CACHE[cache_key]

    if key not in _VECTORS:
        _SYNONYM_CACHE[cache_key] = set()
        return set()

    # Skip the word itself if it's technical
    if _is_technical(key):
        _SYNONYM_CACHE[cache_key] = set()
        return set()

    try:
        neighbors = _VECTORS.most_similar(key, topn=TOP_N)
    except KeyError:
        return set()

    synonyms: set[str] = set()
    for neighbor, similarity in neighbors:
        if similarity < COSINE_THRESHOLD:
            break   # sorted descending, no point continuing

        n = neighbor.lower()

        # Basic form checks
        if not n.isalpha() or len(n) < 4 or n in _WORDNET_SKIP:
            continue

        # Must not be technical/rare
        if _is_technical(n):
            continue

        synonyms.add(n)

    _SYNONYM_CACHE[cache_key] = synonyms
    return synonyms


def degrade_lexical(text: str, level: float, rng: random.Random) -> str:
    """Collapse vocabulary using FastText/GloVe embeddings + wordfreq filtering.

    Synonym candidates come from nearest neighbors in embedding space
    (cosine similarity >= COSINE_THRESHOLD). Technical and rare words
    are excluded via wordfreq frequency threshold so domain jargon,
    acronyms, and proper nouns are never substituted.

    Two-pass algorithm:
      Pass 1 (right-to-left): last occurrence of a synonym group becomes
                               canonical — sits toward END of text.
      Pass 2 (left-to-right): replace earlier synonyms with canonical
                               with probability = level (no hard ceiling,
                               guaranteed gradient across severity levels).
    """
    if level <= 0:
        return text

    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        return text

    _ALLOWED_POS = set(_POS_GROUP.keys())

    # Phase 1: identify eligible content-word positions
    content_indices: list[int] = []

    for i, (word, tag) in enumerate(tagged):
        if (tag in _ALLOWED_POS
                and len(word) >= 4
                and word.isalpha()
                and word.lower() not in _WORDNET_SKIP
                and not _is_technical(word)):   # skip technical terms early
            content_indices.append(i)

    if not content_indices:
        return text

    # Phase 2: RIGHT-TO-LEFT — register canonical forms.
    # Store BASE LEMMA as canonical (not surface form) so _transfer_morphology
    # inflects it correctly in Phase 3 without double-inflection bugs.
    canonical: dict[str, str] = {}   # word.lower() -> canonical BASE LEMMA

    for i in reversed(content_indices):
        word, tag = tagged[i]
        key   = word.lower()
        lemma = _get_lemma(word, tag)   # base form: "called" -> "call"

        if key in canonical:
            continue

        synonyms = _get_embedding_synonyms(key, tag)
        if not synonyms:
            continue

        # Store lemma as canonical so inflection is applied once in Phase 3
        all_group = synonyms | {key}
        for member in all_group:
            if member not in canonical:
                canonical[member] = lemma   # BASE LEMMA, not surface form

    # Phase 3: LEFT-TO-RIGHT — apply replacements probabilistically
    new_tokens = list(tokens)

    # Build sentence boundaries so we can prevent same-sentence duplicates.
    # A sentence ends at '.', '!', '?' tokens.
    sent_id_at: dict[int, int] = {}
    sid = 0
    for i, (w, _) in enumerate(tagged):
        sent_id_at[i] = sid
        if w in (".", "!", "?"):
            sid += 1

    # Track which canonical lemmas have already appeared in each sentence.
    # key: (sentence_id, canonical_lemma) -> True if already present
    canonical_in_sent: set[tuple[int, str]] = set()
    for i, (word, tag) in enumerate(tagged):
        lemma = _get_lemma(word, tag)
        if lemma in canonical.values():
            canonical_in_sent.add((sent_id_at[i], lemma))

    for i in content_indices:
        word, tag = tagged[i]
        key   = word.lower()
        lemma = _get_lemma(word, tag)   # lemma of the word being replaced

        if key not in canonical and lemma not in canonical:
            continue

        # Look up by surface form first, fall back to lemma
        best_lemma = canonical.get(key) or canonical.get(lemma)
        if best_lemma is None:
            continue

        # Skip if canonical IS this word's lemma (nothing to change)
        if best_lemma == lemma:
            continue

        # Skip if the canonical word already appears elsewhere in this sentence
        sent = sent_id_at[i]
        if (sent, best_lemma) in canonical_in_sent:
            continue

        if rng.random() > level:
            continue

        # Inflect the canonical BASE LEMMA to match original word's morphology
        replacement = _transfer_morphology(word, best_lemma, tag)
        if word[0].isupper(): replacement = replacement.capitalize()
        if word.isupper():    replacement = replacement.upper()
        new_tokens[i] = replacement

        # Mark this canonical as now present in this sentence
        canonical_in_sent.add((sent, best_lemma))

    # Phase 4: reconstruct with TreebankWordDetokenizer
    return _DETOKENIZER.detokenize(new_tokens)


# ═══════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════

AXIS_FUNCTIONS = {
    "grammar":     degrade_grammar,
    "coherence":   degrade_coherence,
    "information": degrade_information,
    "lexical":     degrade_lexical,
}


def make_seed(article_title: str, axis: str, level: float, rep: int) -> int:
    """
    FIX 1: Deterministic seed via hashlib.md5.
    Python's built-in hash() is session-randomised (PYTHONHASHSEED) since
    Python 3.3, so identical inputs produce different seeds across runs.
    MD5 is stable across machines, OS versions, and Python versions.
    """
    key = f"{article_title}_{axis}_{level}_{rep}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**31)


def degrade_text(text: str, axis: str, level: float, seed: int = 42) -> str:
    """Apply a single degradation axis at a given level to `text`."""
    rng = random.Random(seed)
    return AXIS_FUNCTIONS[axis](text, level, rng)


# ═══════════════════════════════════════════════════════════════════
# Batch Processing
# ═══════════════════════════════════════════════════════════════════

def run(config: dict, corpus: list[dict]) -> list[dict]:
    """Generate all degraded samples. Returns list of sample dicts."""
    deg_cfg  = config["degradation"]
    levels   = deg_cfg["levels"]
    spl      = deg_cfg["samples_per_level"]
    out_dir  = Path(deg_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "degraded_samples.json"
    if out_file.exists():
        print("[Degradation] Loading existing degraded samples …")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    axes = [a for a, cfg in deg_cfg["axes"].items() if cfg.get("enabled", True)]
    all_samples: list[dict] = []
    sample_id = 0
    total     = len(corpus) * len(axes) * len(levels) * spl

    with tqdm(total=total, desc="Degrading texts", unit="sample",
              dynamic_ncols=True, mininterval=0.5) as pbar:
        for a_idx, article in enumerate(corpus):
            for axis in axes:
                for level in levels:
                    for rep in range(spl):
                        seed = make_seed(article["title"], axis, level, rep)

                        # level 0.0 → original text unchanged (no corruption)
                        if level == 0.0:
                            degrad = article["text"]
                        else:
                            degrad = degrade_text(article["text"], axis, level, seed=seed)

                        all_samples.append({
                            "id":            sample_id,
                            "source_title":  article["title"],
                            "category":      article.get("category", "uncategorized"),
                            "axis":          axis,
                            "level":         level,
                            "repetition":    rep,
                            "seed":          seed,
                            "original_text": article["text"],   # always included
                            "degraded_text": degrad,
                        })
                        sample_id += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "article": f"{a_idx+1}/{len(corpus)}",
                            "axis":    axis,
                            "level":   f"{level:.1f}",
                        }, refresh=False)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"[Degradation] Saved {len(all_samples)} samples → {out_file}")
    return all_samples