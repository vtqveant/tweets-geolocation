from unidecode import unidecode

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



