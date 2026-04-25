import numpy as np

NEUTRAL_C = ['n', 'r', 'l']

V  = [ 'a','e','i','o','u' ]
VV = [ 'ai', 'au', 'oi', 'ei', 'ou',
       'ae', 'ay', 'ey', 'ow', 'oy'  ]

STOP_VOICELESS = [ 'p', 't', 'k' ]
STOP_VOICED = [ 'b', 'd', 'g' ]
STOP = [ *STOP_VOICED, *STOP_VOICELESS ]

FRICATIVES = ['f', 'v', 's', 'z', 'sh', 'th', 'ch', 'j']
NASALS = [ 'm', 'n' ]
LIQUIDS = ['l', 'r']
GLIDE = ['w', 'y']
VOWELS = [ *V, *VV ]

CODA_CC = [
    [STOP_VOICELESS, FRICATIVES],
    [STOP_VOICELESS, NASALS],
    [STOP_VOICELESS, LIQUIDS],
    [FRICATIVES, NASALS],
    [FRICATIVES, LIQUIDS],
    [NASALS, LIQUIDS]
]

CODA_C = [
    *STOP_VOICELESS,
    *FRICATIVES,
    *NASALS,
    *LIQUIDS,
]

ONSET_C = [
    *STOP,
    *FRICATIVES,
    *NASALS,
    *LIQUIDS,
    #*GLIDE
]

ONSET_CC = [
    #[STOP_VOICELESS, STOP_VOICED],
    #[STOP_VOICELESS, FRICATIVES],
    [STOP_VOICELESS, NASALS],
    [STOP_VOICELESS, LIQUIDS],
    #[STOP_VOICELESS, GLIDE],
    #[STOP_VOICED, FRICATIVES],
    [STOP_VOICED, NASALS],
    [STOP_VOICED, LIQUIDS],
    #[STOP_VOICED, GLIDE],
    [FRICATIVES, NASALS],
    [FRICATIVES, LIQUIDS],
    #[FRICATIVES, GLIDE],
    #[NASALS, LIQUIDS],
    #[NASALS, GLIDE],
    #[LIQUIDS, GLIDE],
]

SYLLABLE_TEMPLATES = [
    ('C',  'V'),
    ('V',  'C'),
    ('C',  'VV'),
    ('C',  'V', 'C'),
    ('CC', 'V', 'C'),
]

def onset(rng, p1):
    if p1 == 'CC':
        i = rng.integers(0, len(ONSET_CC))
        cs1, cs2 = ONSET_CC[i]
        return str(rng.choice(cs1) + rng.choice(cs2))
    return str(rng.choice(ONSET_C))

def coda(rng, p1):
    if p1 == 'CC':
        i = rng.integers(0, len(ONSET_CC))
        cs1, cs2 = ONSET_CC[i]
        return str(rng.choice(cs1) + rng.choice(cs2))
    return str(rng.choice(CODA_C))

def vowel(rng, p):
    if p == 'V':
        return str(rng.choice(np.array([*V])))
    elif p == 'VV':
        return str(rng.choice(np.array([*VV])))
    raise Exception(f'Invalid syllable vowel: {p}')

def make_name(seed):
    rng = np.random.default_rng(seed)

    word = []
    tpl:list[list[str]] = []
    n_syllables = rng.integers(2, 4)
    for i in range(n_syllables):
        j = rng.integers(0, len(SYLLABLE_TEMPLATES))
        tpl.append(SYLLABLE_TEMPLATES[j])

    # if tpl[0][0] == 'CC':
    #    tpl.insert(0, ['V'])
    
    # if tpl[-1][-1] == 'CC':
    #    tpl.append(['V'])

    curr = None
    for prev, curr, next in zip([None, *tpl], tpl, [*tpl[1:], None]):
        if prev is not None:
            ch = rng.choice(['', '\'', '-'])
            if ch:
                word.append(str(ch))
        
        if prev is not None and prev[-1] in ('V', 'VV') and curr[0] in ('V', 'VV'):
           word.append(str(rng.choice([*NEUTRAL_C])))
        
        has_nucleus = False
        for p in curr:
            if p in ('C','CC'):
                word.append(onset(rng, p) if has_nucleus else coda(rng, p))
            elif p in ('V', 'VV'):
                has_nucleus = True
                word.append(vowel(rng, p))
            else:
                raise Exception('Invalid template')
    
    # No vowels at the end of a word
    if curr is not None and curr[-1] in ('V', 'VV'):
       word.append(str(rng.choice([*NEUTRAL_C])))

    # print()
    # print(tpl)
    # print(word)
    return "".join(word).title()
