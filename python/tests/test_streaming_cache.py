"""Eviction order of the shared expert LRU."""

from edge0.streaming.cache import SharedExpertCache


def test_put_of_a_held_key_makes_it_newest():
    # sync_actuals re-puts the experts the router just used to promote them;
    # an assignment alone left them in their old slot, first in line to go.
    cache = SharedExpertCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 1)
    cache.put("c", 3)
    assert cache.peek("a") == 1
    assert cache.peek("b") is None
    assert cache.size() == 2


def test_put_and_get_promote_the_same_way():
    by_put = SharedExpertCache(2)
    by_get = SharedExpertCache(2)
    for cache in (by_put, by_get):
        cache.put("a", 1)
        cache.put("b", 2)
    by_put.put("a", 1)
    by_get.get("a")
    for cache in (by_put, by_get):
        cache.put("c", 3)
    assert list(by_put._cache) == list(by_get._cache) == ["a", "c"]
