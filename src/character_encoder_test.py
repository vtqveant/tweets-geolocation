import unittest

import character_encoder
from character_encoder import CharacterEncoder


class CharacterEncoderTest(unittest.TestCase):

    def setUp(self):
        self.encoder = CharacterEncoder(character_encoder.ENCODING_SIZE_SMALL)

    def tearDown(self) -> None:
        self.encoder = None

    def testUnsupportedEncodingSizeRaisesValueError(self):
        self.assertRaises(ValueError, CharacterEncoder, 10)

    def testNumRowsWithPadding(self):
        for text in ['This is English', '谢谢你', u'ko\u017eu\u0161\u010dek']:
            result = self.encoder.encode(text)
            self.assertEquals(character_encoder.NUM_ROWS, len(result))

    def testNumRowsWithTruncation(self):
        for text in [''.zfill(280), ''.zfill(300)]:
            result = self.encoder.encode(text)
            self.assertEquals(character_encoder.NUM_ROWS, len(result))

    def testResultRowSizeEqualsEncodingSize(self):
        for text in ['This is English', '谢谢你', u'ko\u017eu\u0161\u010dek', 'Junto a ti!']:
            result = self.encoder.encode(text)
            for i in range(len(result)):
                self.assertEquals(character_encoder.ENCODING_SIZE_SMALL, len(result[i]))

    def testLatinRowCountEqualsInputLength(self):
        text = "This is English"
        result = self.encoder.encode(text)
        self.assertEquals(len(text), self._countNonPaddingRows(result))

    def testMandarinRowCountNotEqualsInputLength(self):
        text = "谢谢你"
        result = self.encoder.encode(text)
        self.assertNotEquals(len(text), self._countNonPaddingRows(result))

    def testMandarinRowCount(self):
        result = self.encoder.encode("谢谢")
        self.assertEquals(6, self._countNonPaddingRows(result))

    def testLatinDiacriticIsOneRow(self):
        result = self.encoder.encode("\u017e")
        self.assertEquals(1, self._countNonPaddingRows(result))

    def testLatinDiacriticDetected(self):
        result = self.encoder.encode("\u017ea")
        first_row = result[0]
        diacritic_range = self._getRange(first_row, 14, 29)
        self.assertTrue('1' in diacritic_range)
        second_row = result[1]
        diacritic_range = self._getRange(second_row, 14, 29)
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

    @staticmethod
    def _countNonPaddingRows(matrix):
        """count the number of rows which are not zero-padding rows"""
        count = 0
        for row in matrix:
            if '1' in row:
                count += 1
        return count

    def testAwkwardRangeHelperFunction(self):
        self.assertEquals('456', self._getRange('123456789', 4, 6))
