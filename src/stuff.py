
# padding = [''.zfill(self.d)] * (NUM_ROWS - len(rows))

a = b'0001000'
print(a)
print(a[0])

i = float(a[0] - 48)
print(i)

l = [float(i - 48) for i in a]
print(l)


b = b'0011100'
print(b)

c = a + b
print(c)
l2 = [float(i - 48) for i in c]
print(l2)


d = b''.zfill(10)
print(d)
s = d[:3] + b'3' + d[3 + 1:]
print(s)


def _replace(bytestring: bytes, index: int, value: bytes):
    return bytestring[:index] + value + bytestring[index + 1:]


print(_replace(b'0000', 0, b'1'))
print(_replace(b'0000', 1, b'1'))
print(_replace(b'0000', 2, b'1'))
print(_replace(b'0000', 3, b'1'))