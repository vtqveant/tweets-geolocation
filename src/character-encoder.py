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

    def __init__(self, d):
        self.d = d

        # targeting Linux, s. https://stackoverflow.com/questions/7291120/get-unicode-code-point-of-a-character-using-python/42262842#42262842
        # s. also https://pypi.org/project/Unidecode/ for unidecode's requirements to support characters outside
        # Basic Multilingual Plane (BMP) (such as bold, italic, script, mathematical notation etc.)
        if sys.maxunicode <= 0xffff:
            print("WARNING! sys.maxunicode <= 0xffff")

    def encode(self, text):
        # 1. normalize the string's representation using Unicode's NFKC normalization strategy.
        # codepoints = [self._get_wide_ordinal(c) for c in unicodedata.normalize('NFKC', text)]
        # print(codepoints)

        # 2. loop through each code point in the normalized string and update the corresponding rows in the encoding matrix
        # determine codepoint category, s. http://www.unicode.org/reports/tr44/#General_Category_Values
        for character in unicodedata.normalize('NFKC', text):
            category = unicodedata.category(character)
            print(category)
            if not category.startswith('L'):
                print(character)
                print('Using 1 row')

                retval = self._encode_unicode_category(category) + self._encode_case(character, category) + \
                         self._encode_directionality(character)
                print(retval)

            else:
                transliterated = unidecode(character).lower()
                print(transliterated)
                print(f'Using {len(transliterated)} rows')

                for l in transliterated:
                    retval = self._encode_unicode_category(category) + self._encode_case(character, category) + \
                             self._encode_directionality(character)
                    print(retval)

        decoded = [unidecode(c) for c in unicodedata.normalize('NFKC', text)]
        print(decoded)

        return None

    def _get_wide_ordinal(self, c):
        """s. https://stackoverflow.com/questions/7291120/get-unicode-code-point-of-a-character-using-python/7291240#7291240"""
        return ord(c) if len(c) != 2 else 0x10000 + (ord(c[0]) - 0xD800) * 0x400 + (ord(c[1]) - 0xDC00)

    def _encode_unicode_category(self, category):
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

    def _encode_case(self, character, category):
        if category.startswith('L') and (character.isupper() or character.istitle()):
            return '1'
        else:
            return '0'

    def _encode_directionality(self, character):
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


def main():
    encoder = CharacterEncoder(ENCODING_SIZE_SMALL)
    # encoder.encode("понимает по-русски")
    encoder.encode("谢谢你")


if __name__ == '__main__':
    main()
