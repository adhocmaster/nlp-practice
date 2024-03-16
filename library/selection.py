from typing import *
import random

def split(items: Iterable[any], ratio: float) -> Tuple[Iterable[any], Iterable[any]]:
    n = len(items)
    firstN = int(n * ratio)
    firstIndices = set(random.sample(range(n), firstN))
    first = []
    second = []
    for i in range(n):
        if i in firstIndices:
            first.append(items[i])
        else:
            second.append(items[i])
    return first, second
    