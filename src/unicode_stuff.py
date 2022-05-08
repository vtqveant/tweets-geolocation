from unidecode import unidecode
import unicodedata

print(u'ko\u017eu\u0161\u010dek')

# Get transliteration for following
# non-ASCII text (Unicode string)
print(unidecode(u'ko\u017eu\u0161\u010dek'))

# Get transliteration for following
# non-ASCII text (Devanagari)
print(unidecode("आप नीचे अपनी भाषा और इनपुट उपकरण चुनें और लिखना आरंभ करें"))

# Get transliteration for following
# non-ASCII text (Chinese)
print(unidecode("谢谢你"))

# Get transliteration for following
# non-ASCII text (Japanese)
print(unidecode("ありがとう。"))

# Get transliteration for following
# non-ASCII text (Russian)
print(unidecode("улыбаться Владимир Путин"))

retval = unicodedata.decomposition(u'\u017e')
print('1', retval)

renormalized = unicodedata.normalize('NFD', u'\u017e')
print(renormalized)
retval = unicodedata.decomposition(renormalized[0])
print('2', retval)