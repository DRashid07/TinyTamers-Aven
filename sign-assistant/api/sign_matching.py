"""Conservative, local Azerbaijani text matching for the available sign vocabulary.

Owner: D (Direction B/Speech/Eval). See CONTRACT.md.
Exact words and phrases take precedence over explicit aliases and a finite set of
grammatical forms. An unknown word stays visible; matching never uses a prefix.
"""

import re
import unicodedata


# Include combining marks so an unknown decomposed word is retained as typed.
_TOKEN = re.compile(r"[^\W_]+(?:[\u0300-\u036f][^\W_]*)*")
_VOWELS = "aıoueəiöü"
_BACK = "aıou"
_HIGH = {"a": "ı", "ı": "ı", "o": "u", "u": "u",
         "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}

# These are lexical alternations, not a general permission to replace final k/q.
_SOFT_NOUN = {
    "uşaq": "uşağ", "ayaq": "ayağ", "qulaq": "qulağ", "qonaq": "qonağ",
    "otaq": "otağ", "bıçaq": "bıçağ", "bayraq": "bayrağ", "ərzaq": "ərzağ", "çörək": "çörəy",
    "ürək": "ürəy", "çiçək": "çiçəy", "köynək": "köynəy", "inək": "inəy",
    "bilək": "biləy", "böyrək": "böyrəy",
}
_PRONOUNS = {
    "mən": ("məni", "mənim", "mənə", "məndə", "məndən"),
    "sən": ("səni", "sənin", "sənə", "səndə", "səndən"),
    "biz": ("bizi", "bizim", "bizə", "bizdə", "bizdən"),
    "siz": ("sizi", "sizin", "sizə", "sizdə", "sizdən"),
    "o": ("onu", "onun", "ona", "onda", "ondan"),
    "bu": ("bunu", "bunun", "buna", "bunda", "bundan"),
}
_NO_NOUN_FORMS = set(_PRONOUNS) | {
    "mənim", "sənin", "bizim", "sizin", "onun", "bunun", "mənə",
    "necə", "hansı", "var", "yox", "deyil", "burda", "orda", "harda",
    "bura", "ora", "bilər", "bilmir", "edir", "olar", "lazım", "nədir",
    "istəyirəm", "xatırladım", "yaxşı", "pis", "çox", "üçün",
}


def az_lower(text):
    """NFC-normalize and lowercase with Azerbaijani İ/i and I/ı casing."""
    return unicodedata.normalize("NFC", text).replace("İ", "i").replace("I", "ı").lower()


def _harmony(word):
    vowel = next((letter for letter in reversed(word) if letter in _VOWELS), None)
    if vowel is None:
        return None
    return ("a" if vowel in _BACK else "ə", _HIGH[vowel])


def _noun_forms(word):
    """Only singular/plural and five cases, with the expected vowel harmony."""
    harmony = _harmony(word)
    if len(word) < 2 or harmony is None or word in _NO_NOUN_FORMS:
        return set()
    # A lexical possessive ending takes n before every case ending. Applying
    # the ordinary vowel-final rule would invent forms such as ad günüyə.
    if word == "günü":
        return {"gününə", "gününü", "gününün", "günündə", "günündən"}
    low, high = harmony
    vowel_final = word[-1] in _VOWELS
    stem = _SOFT_NOUN.get(word, word)
    # su is the bounded vowel-final exception: suyu/suyun, rather than sunu/sunun.
    object_buffer = "y" if word == "su" else "n" if vowel_final else ""
    cases = {
        stem + ("y" if vowel_final else "") + low,
        stem + object_buffer + high,
        stem + object_buffer + high + "n",
        word + "d" + low,
        word + "d" + low + "n",
    }
    plural = word + "l" + low + "r"
    plural_high = "ı" if low == "a" else "i"
    return cases | {
        plural, plural + low, plural + plural_high, plural + plural_high + "n",
        plural + "d" + low, plural + "d" + low + "n",
    }


def _present_persons(base):
    low, high = _harmony(base)
    consonant = "q" if low == "a" else "k"
    return {
        base, base + low + "m", base + "s" + low + "n", base + high + consonant,
        base + "s" + high + "n" + high + "z", base + "l" + low + "r",
    }


def _past_persons(base):
    low, high = _harmony(base)
    return {
        base, base + "m", base + "n", base + ("q" if low == "a" else "k"),
        base + "n" + high + "z", base + "l" + low + "r",
    }


def _future_persons(base):
    low, high = _harmony(base)
    soft = base[:-1] + ("ğ" if base[-1] == "q" else "y")
    return {
        base, soft + low + "m", base + "s" + low + "n",
        soft + high + ("q" if low == "a" else "k"),
        base + "s" + high + "n" + high + "z", base + "l" + low + "r",
    }


def _verb_forms(infinitive):
    """Generate common positive/negative present, past, future and imperative forms."""
    root = infinitive[:-3]
    harmony = _harmony(root)
    if not root or harmony is None:
        return set(), set()
    low, high = harmony
    # get -> gedir/gedəcək; getdim and getmə retain t.
    present_root = "ged" if root == "get" else root
    if present_root[-1] in _VOWELS:
        present_root += "y"
    present = present_root + high + "r"
    future_root = "ged" if root == "get" else root
    if future_root[-1] in _VOWELS:
        future_root += "y"
    # ye -> yeyəcək follows the same vowel-final rule.
    future = future_root + low + "c" + low + ("q" if low == "a" else "k")
    positive = {root} | _present_persons(present) | _past_persons(root + "d" + high)
    positive |= _future_persons(future)
    # The infinitive also has valid case forms: getməyə, almağa, yeməkdən.
    inf_low, inf_high = _harmony(infinitive)
    inf_soft = infinitive[:-1] + ("ğ" if infinitive[-1] == "q" else "y")
    positive |= {
        inf_soft + inf_low, inf_soft + inf_high, inf_soft + inf_high + "n",
        infinitive + "d" + inf_low, infinitive + "d" + inf_low + "n",
    }
    neg_root = root + "m" + low
    negative = {neg_root, neg_root + infinitive[-3:]}
    negative |= _present_persons(root + "m" + high + "r")
    negative |= _past_persons(neg_root + "d" + ("ı" if low == "a" else "i"))
    neg_future = neg_root + "y" + low + "c" + low + ("q" if low == "a" else "k")
    negative |= _future_persons(neg_future)
    return positive, negative


def _add(forms, words, sequence):
    """Ambiguous grammatical/alias matches stay unknown instead of choosing an id."""
    key = tuple(words)
    if key not in forms:
        forms[key] = sequence
    elif forms[key] != sequence:
        forms[key] = None


def _build_forms(vocab):
    exact, aliases, grammatical = {}, {}, {}
    for entry in vocab:
        words = tuple(az_lower(word) for word in _TOKEN.findall(entry["az"]))
        if words:
            exact.setdefault(words, [("sign", entry["id"])])

    def alias(source, *targets):
        sequences = [exact.get(tuple(_TOKEN.findall(az_lower(target)))) for target in targets]
        if all(sequences):
            sequence = [item for items in sequences for item in items]
            _add(aliases, _TOKEN.findall(az_lower(source)), sequence)

    for source, targets in {
        "necəsiniz": ("necə", "siz"), "necəsiz": ("necə", "siz"),
        "necə siniz": ("necə", "siz"), "necəsən": ("necə", "sən"),
        "bugün": ("bu gün",), "burada": ("burda",), "buradan": ("burda",),
        "harada": ("harda",), "orada": ("orda",), "oradan": ("orda",),
    }.items():
        alias(source, *targets)
    for source in ("yaxşıyam", "yaxşısan", "yaxşıdır", "yaxşıyıq", "yaxşısınız", "yaxşıdırlar"):
        alias(source, "yaxşı")
    for source in ("deyil", "deyiləm", "deyilsən", "deyilik", "deyilsiniz", "deyillər", "yoxdur"):
        alias(source, "yox")
    # These recorded predicates have bounded personal forms. The negative
    # bilmir class already carries negation and needs no extra yox.
    for predicate in ("bilmir", "bilər", "edir", "olar"):
        if (predicate,) in exact:
            for word in _present_persons(predicate):
                alias(word, predicate)

    negation = exact.get(("yox",))
    for words, sequence in exact.items():
        last = words[-1]
        if last.endswith(("maq", "mək")):
            positive, negative = _verb_forms(last)
            for form in positive:
                _add(grammatical, (*words[:-1], form), sequence)
            if negation:
                for form in negative:
                    _add(grammatical, (*words[:-1], form), sequence + negation)
        else:
            for form in _PRONOUNS.get(last, ()):
                _add(grammatical, (*words[:-1], form), sequence)
            for form in _noun_forms(last):
                _add(grammatical, (*words[:-1], form), sequence)
    return exact, aliases, grammatical


def from_fallback(text, vocab):
    """Return ordered (kind, id/word) tuples, keeping unrecognized tokens as typed.

    Match the longest available phrase. For the same phrase length a canonical
    vocabulary entry wins over an explicit alias, which wins over inflection.
    Negative inflections require a recorded negative class or a yox sign; without
    either, the original negative word remains out of vocabulary.
    """
    tokens = _TOKEN.findall(text)
    lowered = [az_lower(token) for token in tokens]
    layers = _build_forms(vocab)
    lengths = sorted({len(words) for layer in layers for words in layer}, reverse=True)
    result, index = [], 0
    while index < len(tokens):
        matched = False
        for length in lengths:
            if index + length > len(tokens):
                continue
            phrase = tuple(lowered[index:index + length])
            sequence = next((layer[phrase] for layer in layers if layer.get(phrase)), None)
            if sequence:
                result.extend(sequence)
                index += length
                matched = True
                break
        if not matched:
            result.append(("oov", tokens[index]))
            index += 1
    return result
