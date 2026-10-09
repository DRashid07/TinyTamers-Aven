"""Behavioral tests for conservative Azerbaijani text matching.

Owner: D (Direction B/Speech/Eval).
"""

import unicodedata

import pytest

from api.sign_matching import az_lower, from_fallback


def vocabulary(*entries):
    return [{"id": identifier, "az": word} for identifier, word in entries]


VOCAB = vocabulary(
    ("salam", "salam"), ("nece", "necə"), ("siz", "siz"), ("sen", "sən"),
    ("men", "mən"), ("menim", "mənim"), ("bu", "bu"), ("bu_gun", "bu gün"),
    ("ev", "ev"), ("hekim", "həkim"), ("usaq", "uşaq"), ("sabah", "sabah"),
    ("getmek", "getmək"), ("gelmek", "gəlmək"), ("istemek", "istəmək"),
    ("almaq", "almaq"), ("yemek", "yemək"), ("olmaq", "olmaq"),
    ("oxumaq", "oxumaq"), ("islemek", "işləmək"), ("gormek", "görmək"),
    ("burda", "burda"), ("harda", "harda"), ("yaxsi", "yaxşı"),
)


def signs(*ids):
    return [("sign", identifier) for identifier in ids]


@pytest.mark.parametrize("text, expected", [
    ("SALAM, NECƏSİNİZ?", signs("salam", "nece", "siz")),
    ("Salam! Necəsən?", signs("salam", "nece", "sen")),
    ("necəsiz / necə siniz", signs("nece", "siz", "nece", "siz")),
    ("Mən yaxşıyam.", signs("men", "yaxsi")),
    ("Bugün burada, buradan harada?", signs("bu_gun", "burda", "burda", "harda")),
    ("Bu gün evə", signs("bu_gun", "ev")),
])
def test_greetings_and_explicit_common_aliases(text, expected):
    assert from_fallback(text, VOCAB) == expected


def test_greeting_alias_requires_every_sign():
    assert from_fallback("necəsiniz", vocabulary(("nece", "necə"))) == [("oov", "necəsiniz")]
    assert from_fallback("necəsən", vocabulary(("siz", "siz"), ("nece", "necə"))) == [("oov", "necəsən")]


@pytest.mark.parametrize("text, expected", [
    ("Sabah həkimə gedirəm.", signs("sabah", "hekim", "getmek")),
    ("getdim gedirsən gedirik gedirsiniz gedirlər", signs(*["getmek"] * 5)),
    ("gəlirəm gələcəyəm gəldik", signs(*["gelmek"] * 3)),
    ("istəyirəm istəyirsiniz istədim", signs(*["istemek"] * 3)),
    ("alıram alacaqsınız aldıq", signs(*["almaq"] * 3)),
    ("yeyirəm yedim yeyəcəyik", signs(*["yemek"] * 3)),
    ("oluruq oxuyursunuz işləyirlər görürəm", signs("olmaq", "oxumaq", "islemek", "gormek")),
    ("getməyə almağa yeməkdən", signs("getmek", "almaq", "yemek")),
])
def test_common_positive_verb_forms(text, expected):
    assert from_fallback(text, VOCAB) == expected


@pytest.mark.parametrize("text", ["getmirəm", "istəmirəm", "almadım", "yeməyəcəyəm", "gəlmə", "getməmək"])
def test_negation_is_preserved_without_a_negation_sign(text):
    assert from_fallback(text, VOCAB) == [("oov", text)]


@pytest.mark.parametrize("text, identifier", [
    ("getmirəm", "getmek"), ("istəmirəm", "istemek"), ("almadım", "almaq"),
    ("yeməyəcəyəm", "yemek"), ("gəlmə", "gelmek"), ("getməmək", "getmek"),
])
def test_negated_verb_requires_a_following_negation_sign(text, identifier):
    assert from_fallback(text, VOCAB + vocabulary(("yox", "yox"))) == signs(identifier, "yox")


def test_known_negative_class_is_not_changed_to_an_affirmative_class():
    vocab = VOCAB + vocabulary(("bilmir", "bilmir"), ("bilmek", "bilmək"), ("yox", "yox"))
    assert from_fallback("bilmirəm bilmirsiniz bilmir", vocab) == signs("bilmir", "bilmir", "bilmir")


def test_validated_noun_cases_and_plurals_include_short_words():
    assert from_fallback("evə evdə evdən evin evləri həkimə həkimlərin uşağa uşaqlar", VOCAB) == signs(
        "ev", "ev", "ev", "ev", "ev", "hekim", "hekim", "usaq", "usaq"
    )
    assert from_fallback("Mənim mənə məndən sizə", VOCAB) == signs("menim", "men", "men", "siz")


@pytest.mark.parametrize("word", [
    "mənzil", "salamander", "sabahlıq", "yoxlama", "uşaqa", "həkima", "evdəki", "getdir",
    "istəyirəmsə", "evlilik", "necəlik", "almanca", "Nərgiz", "12345",
])
def test_prefixes_derivations_and_invalid_endings_do_not_invent_signs(word):
    assert from_fallback(word, VOCAB + vocabulary(("yox", "yox"))) == [("oov", word)]


def test_unknown_words_and_negation_keep_their_original_order_and_case():
    assert from_fallback("Salam, Nərgiz! Evə getmirəm.", VOCAB) == [
        ("sign", "salam"), ("oov", "Nərgiz"), ("sign", "ev"), ("oov", "getmirəm")
    ]


def test_specific_canonical_entries_win_over_inflection_and_aliases():
    vocab = VOCAB + vocabulary(
        ("greeting", "necəsiniz"), ("specific_want", "istəyirəm"), ("mene", "mənə")
    )
    assert from_fallback("Necəsiniz istəyirəm mənə", vocab) == signs("greeting", "specific_want", "mene")


def test_exact_phrase_and_validated_inflected_phrase_win_over_shorter_words():
    assert from_fallback("bu gün bu günə", VOCAB) == signs("bu_gun", "bu_gun")
    assert from_fallback("BU-GÜN", VOCAB) == signs("bu_gun")


def test_azerbaijani_case_and_unicode_normalization():
    assert az_lower("İSTƏMƏK BAKI") == "istəmək bakı"
    decomposed = unicodedata.normalize("NFD", "GÖRÜRƏM İSTƏYİRƏM")
    assert from_fallback(decomposed, VOCAB) == signs("gormek", "istemek")
    unknown = unicodedata.normalize("NFD", "Nərgiz")
    assert from_fallback(unknown, VOCAB) == [("oov", unknown)]


def test_single_letter_classes_are_exact_only_and_empty_vocabulary_is_safe():
    assert from_fallback("d dən", vocabulary(("d", "d"))) == [("sign", "d"), ("oov", "dən")]
    assert from_fallback("Salam!", []) == [("oov", "Salam")]


def test_lexical_noun_case_exceptions_use_valid_buffer_consonants():
    vocab = vocabulary(("su", "su"), ("ad_gunu", "ad günü"), ("erzaq", "ərzaq"))
    assert from_fallback("suya suyu suyun suda sudan sular", vocab) == signs(*["su"] * 6)
    assert from_fallback("ad gününə ad gününü ad gününün ad günündə ad günündən", vocab) == signs(
        *["ad_gunu"] * 5
    )
    assert from_fallback("ərzağa ərzağı ərzağın ərzaqda", vocab) == signs(*["erzaq"] * 4)
    for invalid in ("sunu", "sunun", "günüyə", "günününi", "ərzaqa", "ərzaqı"):
        assert from_fallback(invalid, vocab) == [("oov", invalid)]
    assert from_fallback("ad günüyə", vocab) == [("oov", "ad"), ("oov", "günüyə")]


def test_recorded_predicates_use_personal_forms_and_never_noun_endings():
    vocab = vocabulary(("edir", "edir"), ("olar", "olar"), ("biler", "bilər"),
                       ("lazim", "lazım"), ("nedir", "nədir"), ("isteyirem", "istəyirəm"))
    assert from_fallback("edirəm edirsiniz olaram olarıq bilərəm bilərik", vocab) == signs(
        "edir", "edir", "olar", "olar", "biler", "biler"
    )
    for invalid in ("edirin", "oların", "lazımın", "nədirə", "istəyirəmə"):
        assert from_fallback(invalid, vocab) == [("oov", invalid)]


def test_nominal_negation_stays_local_when_a_negation_sign_exists():
    vocab = VOCAB + vocabulary(("yox", "yox"))
    assert from_fallback("Ev deyil deyiləm deyilsən deyilik deyilsiniz deyillər yoxdur", vocab) == signs(
        "ev", *["yox"] * 7
    )
    assert from_fallback("deyil", vocab + vocabulary(("deyil", "deyil"))) == signs("deyil")


def test_nominal_negation_remains_oov_without_a_negation_sign():
    words = ["deyil", "deyiləm", "deyilsən", "deyilik", "deyilsiniz", "deyillər", "yoxdur"]
    assert from_fallback("Ev " + " ".join(words), VOCAB) == [("sign", "ev")] + [
        ("oov", word) for word in words
    ]
