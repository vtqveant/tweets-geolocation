"""
A cache for individual rows in character encoder with a least frequently used replacement policy
with an internal representation based on conversion to a hex string.
"""

from bitarray import bitarray
from bitarray.util import hex2ba, ba2hex
from cachetools import LFUCache


class CharacterEncodingCache:
    def __init__(self, maxsize):
        self._cache = LFUCache(maxsize=maxsize)

    def put(self, key, row):
        self._cache[key] = self._serialize(row)

    def get(self, key):
        if key in self._cache.keys():
            return self._deserialize(self._cache.get(key))
        else:
            return None

    def _serialize(self, row):
        """
        Takes a UnicodeCNN encoded data represented as a list of bytes, returns a string representation in hex.
        To be representable as hex, a bitarray length must be a multiple of 4 (which is satisfied in UnicodeCNN)
        """
        return ba2hex(bitarray(row.decode('UTF-8')))

    def _deserialize(self, s):
        """
        Takes a string representation of a character encoding, returns a list of bytes
        """
        return hex2ba(s).to01().encode('UTF-8')

