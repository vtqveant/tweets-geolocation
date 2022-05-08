import unittest

import character_encoder
from character_encoder import CharacterEncoder


class CharacterEncoderTest(unittest.TestCase):

    def setUp(self):
        self.encoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)

    def tearDown(self) -> None:
        self.encoder = None

    def testSmth(self):
        result = self.encoder.encode('a')
        print(result)

    def testUnsupportedEncodingSizeRaisesValueError(self):
        self.assertRaises(ValueError, CharacterEncoder, 10)

    def testResultRowSizeEqualsEncodingSize(self):
        result = self.encoder.encode('a')
        self.assertEquals(character_encoder.ENCODING_SIZE_SMALL, len(result[0]))

    def testLatinRowCountEqualsInputLength(self):
        text = "This is English"
        result = self.encoder.encode(text)
        self.assertEquals(len(text), len(result))

    def testMandarinRowCountNotEqualInputLength(self):
        text = "谢谢你"
        result = self.encoder.encode(text)
        self.assertNotEquals(len(text), len(result))

    def testMandarinRowCount(self):
        result = self.encoder.encode("谢谢")
        self.assertEquals(6, len(result))

    def testLatinDiacriticIsOneRow(self):
        result = self.encoder.encode("\u017e")
        self.assertEquals(1, len(result))

    def testLatinDiacriticDetected(self):
        result = self.encoder.encode("\u017ea")
        first_row = result[0]
        diacritic_range = first_row[13:29]
        self.assertTrue('1' in diacritic_range)
        second_row = result[1]
        diacritic_range = second_row[13:29]
        self.assertFalse('1' in diacritic_range)

    def testNoDiacriticInNonLatin(self):
        result = self.encoder.encode("谢谢")
        first_row = result[0]
        diacritic_range = self._getRange(first_row, 14, 29)
        self.assertFalse('1' in diacritic_range)
        second_row = result[1]
        diacritic_range = self._getRange(second_row, 14, 29)
        self.assertFalse('1' in diacritic_range)

    @staticmethod
    def _getRange(row: str, left_index_included: int, right_index_included: int):
        """returns a range for a 1-based indexation, both indices included"""
        return row[left_index_included - 1:right_index_included]

    def testAwkwardRangeHelperFunction(self):
        self.assertEquals('456', self._getRange('123456789', 4, 6))
