from ex1 import ex1

text = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
nt = ex1.normalize_text(text)  # lower casing, padding punctuation with white spaces
print(nt)
lm = ex1.Ngram_Language_Model(n=3, chars=False)
char = lm.char_Ngram(nt, n=1)
lm.build_model(nt)  # *
x = lm.get_model()
y = {'a cat sat': 1, 'cat sat on': 2, 'sat on the': 5, 'on the mat': 4, 'the mat .': 4, 'mat . a': 2, '. a fat': 1, 'a fat cat': 1, 'fat cat sat': 1, '. a rat': 1, 'a rat sat': 1, 'rat sat on': 2, 'mat . the': 1, '. the rat': 1, 'the rat sat': 1, 'on the cat': 2, 'the cat .': 1, 'cat . a': 1, '. a bat': 1, 'a bat spat': 1, 'bat spat on': 1, 'spat on the': 1, 'on the rat': 1, 'the rat that': 1, 'rat that sat': 1, 'that sat on': 1, 'the cat on': 1, 'cat on the': 1}
print(x==y)
print(lm.get_model())  # *
t = lm.generate(context='a cat', n=30)
for e in [t, 'a cat sat on the mat', 'the rat sat on the cat']:  # *
    print('%s | %.3f' % (e, lm.evaluate(e)))