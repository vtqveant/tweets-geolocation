import sys
import unicodedata
from unidecode import unidecode

ENCODING_SIZE_SMALL = 128
ENCODING_SIZE_LARGE = 128
ENCODING_SIZE_HUGE = 256
NUM_ROWS = 280


class CharacterEncoder:
    """
    The character encoder converts the input text into a binary matrix.
    The character encoder produces a 280 by d encoding matrix, where each entry is either 0 or 1,
    and d is a hyperparameter called the encoding size.

    I use a Python 3 bytestring as a representation of the encoding
    A bytestring `b` consisting of zeros and ones can be converted to a list of floats with [float(i - 48) for i in b]
    """

    # s. https://planetmath.org/goodhashtableprimes
    LARGE_PRIME = 98317

    def __init__(self, d):
        if d not in [ENCODING_SIZE_SMALL, ENCODING_SIZE_LARGE, ENCODING_SIZE_HUGE]:
            raise ValueError("Unsupported encoding size")

        # targeting Linux, s. https://stackoverflow.com/questions/7291120/get-unicode-code-point-of-a-character-using-python/42262842#42262842
        # s. also https://pypi.org/project/Unidecode/ for unidecode's requirements to support characters outside
        # Basic Multilingual Plane (BMP) (such as bold, italic, script, mathematical notation etc.)
        if sys.maxunicode <= 0xffff:
            print("WARNING: sys.maxunicode <= 0xffff")

        self.d = d

    def encode(self, text):
        rows = []

        # loop through each code point in the normalized string and update the corresponding rows in the encoding matrix
        for character in unicodedata.normalize('NFKC', text):
            # determine codepoint category, s. http://www.unicode.org/reports/tr44/#General_Category_Values
            category = unicodedata.category(character)

            transliteration = unidecode(character).lower()
            length = len(transliteration)
            for i in range(length):
                first_letter = (i == 0)
                row = self._encode(category, character, transliteration[i], first_letter)
                rows.append(row)

        # make sure the matrix has NUM_ROWS
        if len(rows) > NUM_ROWS:
            del rows[NUM_ROWS:]
        else:
            padding = [b''.zfill(self.d)] * (NUM_ROWS - len(rows))
            rows.extend(padding)
        return rows

    def _encode(self, category, character, transliteration, first_letter):
        retval = self._encode_unicode_category(category)  # columns 1-7
        retval += self._encode_case(character, category)  # column 8
        retval += self._encode_directionality(character)  # columns 9-13
        retval += self._encode_diacritics(character)  # columns 14-29
        retval += self._encode_is_sharp(character)  # column 30
        retval += self._encode_is_at(character)  # column 31
        retval += self._encode_transliteration(transliteration)  # columns 32-57

        latin = self._is_latin(character)
        retval += b'1' if latin else b'0'  # column 58

        retval += b'1' if first_letter else b'0'  # column 59

        num_buckets = self.d - 60
        if latin:
            retval += b''.zfill(num_buckets + 1)  # columns 60-d
        else:
            retval += self._encode_non_latin(character, num_buckets)  # columns 60-d
        return retval

    @staticmethod
    def _is_latin(character):
        return 'LATIN' in unicodedata.name(character)

    @staticmethod
    def _encode_unicode_category(category):
        """
        Letter = Lu | Ll | Lt | Lm | Lo
        Mark = Mn | Mc | Me
        Number = Nd | Nl | No
        Punctuation = Pc | Pd | Ps | Pe | Pi | Pf | Po
        Symbol = Sm | Sc | Sk | So
        Separator = Zs | Zl | Zp
        Other = Cc | Cf | Cs | Co | Cn
        """
        if category.startswith('L'):
            return b'1000000'
        elif category.startswith('M'):
            return b'0100000'
        elif category.startswith('N'):
            return b'0010000'
        elif category.startswith('P'):
            return b'0001000'
        elif category.startswith('S'):
            return b'0000100'
        elif category.startswith('Z'):
            return b'0000010'
        else:
            return b'0000001'

    @staticmethod
    def _encode_case(character, category):
        return b'1' if category.startswith('L') and (character.isupper() or character.istitle()) else b'0'

    @staticmethod
    def _encode_directionality(character):
        """
        s. http://www.unicode.org/reports/tr44/#Bidi_Class_Values

        directionality is one-hot encoded with fields as follows:
            strongly left-to-right
            strongly right-to-left
            weak
            neutral
            explicit formatting command
        """
        bidi_class = unicodedata.bidirectional(character)
        if bidi_class in ['L', '']:
            return b'10000'
        elif bidi_class in ['R', 'AL']:
            return b'01000'
        elif bidi_class in ['EN', 'ES', 'ET', 'AN', 'CS', 'NSM', 'BN']:
            return b'00100'
        elif bidi_class in ['B', 'S', 'WS', 'ON']:
            return b'00010'
        elif bidi_class in ['LRE', 'LRO', 'RLE', 'RLO', 'PDF', 'LRI', 'RLI', 'FSI', 'PDI']:
            return b'00001'

    @staticmethod
    def _encode_diacritics(character):
        """Encode diacritic marks on a character (a form of feature hashing at the mark level)"""
        zeros = b''.zfill(16)
        decomp = unicodedata.decomposition(character)
        if decomp == "":
            return zeros
        else:
            decimals = [int(h, 16) for h in decomp.split(' ')]
            tmp = zeros
            for d in decimals:
                place = (d * CharacterEncoder.LARGE_PRIME) % 16
                tmp = CharacterEncoder._replace(tmp, place, b'1')
            return tmp

    @staticmethod
    def _encode_is_sharp(character):
        return b'1' if character == '#' else b'0'

    @staticmethod
    def _encode_is_at(character):
        return b'1' if character == '@' else b'0'

    @staticmethod
    def _encode_transliteration(character):
        """the Latin transliteration of the character"""
        zeros = b''.zfill(26)
        place = ord(character) - 97  # ord('a') == 97
        if place < 0 or place > 25:
            return zeros
        else:
            return CharacterEncoder._replace(zeros, place, b'1')

    @staticmethod
    def _encode_non_latin(character, num_buckets):
        """Encode a non-Latin character (a form of feature hashing at the character level)"""
        zeros = b''.zfill(num_buckets + 1)
        place = (ord(character) * CharacterEncoder.LARGE_PRIME) % num_buckets
        return CharacterEncoder._replace(zeros, place, b'1')

    @staticmethod
    def _replace(bytestring: bytes, index: int, value: bytes):
        return bytestring[:index] + value + bytestring[index + 1:]


def main():
    encoder = CharacterEncoder(ENCODING_SIZE_SMALL)
    # matrix = encoder.encode("понимает по-русски")
    matrix = encoder.encode("谢谢你")
    # matrix = encoder.encode("This is English")
    # matrix = encoder.encode(u'ko\u017eu\u0161\u010dek')
    # matrix = encoder.encode(' ')
    print(matrix)


if __name__ == '__main__':
    main()
