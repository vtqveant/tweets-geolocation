import sys
import unicodedata
from unidecode import unidecode

ENCODING_SIZE_SMALL = 128
ENCODING_SIZE_LARGE = 128
ENCODING_SIZE_HUGE = 256


class CharacterEncoder:
    """
    The character encoder converts the input text into a binary matrix.
    The character encoder produces a 280 by d encoding matrix, where each entry is either 0 or 1,
    and d is a hyperparameter called the encoding size.

    The intended use of this class is to dump the encoded value to a file, so the generated encoding is a string.
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
            print(category)
            if not category.startswith('L'):
                print(character)
                print('Using 1 row')

                row = self._encode(category, character)
                rows.append(row)
                print(row)

            else:
                transliterated = unidecode(character).strip().lower()
                print(transliterated)
                print(f'Using {len(transliterated)} rows')

                flag = True
                for l in transliterated:
                    row = self._encode(category, character, l, flag)
                    flag = False
                    rows.append(row)
                    print(row)

        decoded = [unidecode(c) for c in unicodedata.normalize('NFKC', text)]
        print(decoded)

        return rows

    def _encode(self, category, character, transliteration=None, is_first_letter=False):
        retval = self._encode_unicode_category(category)  # columns 1-7
        retval += self._encode_case(character, category)  # column 8
        retval += self._encode_directionality(character)  # columns 9-13
        retval += self._encode_diacritics(character)  # columns 14-29
        retval += self._encode_is_sharp(character)  # column 30
        retval += self._encode_is_at(character)  # column 31
        if transliteration is None:
            retval += self._encode_latin(character)  # columns 32-57
            retval += '1'  # column 58
        else:
            retval += self._encode_latin(transliteration)  # columns 32-57
            retval += '0'  # column 58

        retval += '1' if is_first_letter else '0'  # column 59

        length = self.d - 60
        if transliteration is None:
            retval += ''.zfill(length)  # columns 60-d
        else:
            retval += self._encode_non_latin(character, length)  # columns 60-d
        return retval

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
            return '1000000'
        elif category.startswith('M'):
            return '0100000'
        elif category.startswith('N'):
            return '0010000'
        elif category.startswith('P'):
            return '0001000'
        elif category.startswith('S'):
            return '0000100'
        elif category.startswith('Z'):
            return '0000010'
        else:
            return '0000001'

    @staticmethod
    def _encode_case(character, category):
        return '1' if category.startswith('L') and (character.isupper() or character.istitle()) else '0'

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
            return '10000'
        elif bidi_class in ['R', 'AL']:
            return '01000'
        elif bidi_class in ['EN', 'ES', 'ET', 'AN', 'CS', 'NSM', 'BN']:
            return '00100'
        elif bidi_class in ['B', 'S', 'WS', 'ON']:
            return '00010'
        elif bidi_class in ['LRE', 'LRO', 'RLE', 'RLO', 'PDF', 'LRI', 'RLI', 'FSI', 'PDI']:
            return '00001'

    @staticmethod
    def _encode_diacritics(character):
        """Encode diacritic marks on a character (a form of feature hashing at the mark level)"""
        zeros = ''.zfill(16)
        decomp = unicodedata.decomposition(character)
        if decomp == "":
            return zeros
        else:
            decimals = [int(h, 16) for h in decomp.split(' ')]
            tmp = list(zeros)
            for d in decimals:
                tmp[(d * CharacterEncoder.LARGE_PRIME) % 16] = '1'
            return ''.join(tmp)

    @staticmethod
    def _encode_is_sharp(character):
        return '1' if character == '#' else '0'

    @staticmethod
    def _encode_is_at(character):
        return '1' if character == '@' else '0'

    @staticmethod
    def _encode_latin(character):
        """the Latin transliteration of the character"""
        zeros = ''.zfill(26)
        place = ord(character) - 97  # ord('a') == 97
        if place < 0 or place > 25:
            return zeros
        else:
            tmp = list(zeros)
            tmp[place] = '1'
            return ''.join(tmp)

    @staticmethod
    def _encode_non_latin(character, length):
        """Encode a non-Latin character (a form of feature hashing at the character level)"""
        place = (ord(character) * CharacterEncoder.LARGE_PRIME) % length
        tmp = ['0'] * (length + 1)
        tmp[place] = '1'
        return ''.join(tmp)


def main():
    encoder = CharacterEncoder(ENCODING_SIZE_SMALL)
    # encoder.encode("понимает по-русски")
    encoder.encode("谢谢你")
    # encoder.encode("This is English")
    # encoder.encode(u'ko\u017eu\u0161\u010dek')


if __name__ == '__main__':
    main()
