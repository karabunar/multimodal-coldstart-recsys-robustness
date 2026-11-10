"""Give every item number of the distributed Baby split its ASIN by aligning interaction graphs.

usage: python -I align_items_to_asin.py <folder with train/val/test.txt> <Baby_5.json (2014 5-core reviews)> <out item_asin.txt>

Both files describe the same user-item graph (160,792 interactions, 19,445 users, 7,050 items). Colour refinement on the
bipartite graph (start: degree; each round: own colour plus the sorted colours of the neighbours) gives a unique colour to
every item; items with equal colour in both graphs are the same item. The result is checked edge by edge for the users that
are uniquely coloured. Output lines: `<asin> <item number>`. Only .txt and .json are read, as data.
"""
import collections
import json
import sys

split_dir, reviews, out = sys.argv[1:4]
ROUNDS = 3


def graph_from_split():
    users, items = collections.defaultdict(set), collections.defaultdict(set)
    for name in ('train.txt', 'val.txt', 'test.txt'):
        for line in open(f'{split_dir}/{name}'):
            t = line.split()
            for i in t[1:]:
                users[int(t[0])].add(int(i))
                items[int(i)].add(int(t[0]))
    return users, items


def graph_from_reviews():
    users, items = collections.defaultdict(set), collections.defaultdict(set)
    for line in open(reviews):
        d = json.loads(line)
        users[d['reviewerID']].add(d['asin'])
        items[d['asin']].add(d['reviewerID'])
    return users, items


palette = {}


def colour(key):
    return palette.setdefault(key, len(palette))


def refine(users, items):
    cu = {u: colour(('u', len(users[u]))) for u in users}
    ci = {i: colour(('i', len(items[i]))) for i in items}
    for _ in range(ROUNDS):
        cu, ci = ({u: colour(('u', cu[u], tuple(sorted(ci[i] for i in users[u])))) for u in users},
                  {i: colour(('i', ci[i], tuple(sorted(cu[u] for u in items[i])))) for i in items})
    return cu, ci


ua, ia = graph_from_split()
ub, ib = graph_from_reviews()
assert (len(ua), len(ia), len(ub), len(ib)) == (19445, 7050, 19445, 7050), 'not the same sizes'
cua, cia = refine(ua, ia)
cub, cib = refine(ub, ib)
count_a, count_b = collections.Counter(cia.values()), collections.Counter(cib.values())
by_colour_a, by_colour_b = {c: i for i, c in cia.items()}, {c: i for i, c in cib.items()}
item_map = {by_colour_a[c]: by_colour_b[c] for c in count_a if count_a[c] == 1 and count_b.get(c) == 1}
user_a, user_b = collections.Counter(cua.values()), collections.Counter(cub.values())
ua_of, ub_of = {c: u for u, c in cua.items()}, {c: u for u, c in cub.items()}
user_map = {ua_of[c]: ub_of[c] for c in user_a if user_a[c] == 1 and user_b.get(c) == 1}
mismatch = sum({item_map[i] for i in ua[u]} != ub[user_map[u]] for u in user_map)
print(f'items mapped {len(item_map)} of {len(ia)}; users uniquely matched {len(user_map)}; edge-set mismatches {mismatch}')
assert len(item_map) == len(ia) and mismatch == 0
with open(out, 'w') as f:
    for number in sorted(item_map):
        f.write(f'{item_map[number]} {number}\n')
