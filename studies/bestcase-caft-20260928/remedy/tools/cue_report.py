"""Surface cues in contrastive pairs: words and bigrams that sit mostly on one side, plus length imbalance.

A mean-difference direction partly encodes any systematic surface difference between the (+) and (-) sides. The
readout is the final token, which integrates the whole sentence, so an asymmetry anywhere matters. The final words
matter most.

Reported per file:
- n-grams (1-2 words) on at least 12 more pairs of one side than of the other;
- final-3-word patterns shared by at least 8 pairs on one side;
- the share of pairs where (+) is longer.
"""
import json
import re
import sys
from collections import Counter


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def grams(ws):
    return set(ws) | {' '.join(ws[i:i + 2]) for i in range(len(ws) - 1)}


def report(path, gap=12):
    rows = [json.loads(l) for l in open(path)]
    pos, neg = Counter(), Counter()
    endp, endn = Counter(), Counter()
    longer = 0
    for r in rows:
        wp, wn = words(r['pos']), words(r['neg'])
        pos.update(grams(wp)); neg.update(grams(wn))
        endp[' '.join(wp[-3:])] += 1; endn[' '.join(wn[-3:])] += 1
        longer += len(wp) > len(wn)
    cues = sorted(((g, pos[g], neg[g]) for g in set(pos) | set(neg) if abs(pos[g] - neg[g]) >= gap),
                  key=lambda x: -abs(x[1] - x[2]))
    ends = [(e, c, '+') for e, c in endp.items() if c >= 8] + [(e, c, '-') for e, c in endn.items() if c >= 8]
    return {'file': path, 'pairs': len(rows), 'pos_longer_share': round(longer / len(rows), 2),
            'one_sided_ngrams': [f'{g!r}: +{p} / -{n}' for g, p, n in cues[:15]], 'repeated_endings': ends}


if __name__ == '__main__':
    bad = False
    for p in sys.argv[1:]:
        r = report(p)
        print(json.dumps(r, indent=1))
        bad |= bool(r['one_sided_ngrams']) or bool(r['repeated_endings'])
    sys.exit(1 if bad else 0)
