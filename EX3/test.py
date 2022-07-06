import EX3.ex3_312237803

model = EX3.ex3_312237803.load_best_model()
# trained_model = EX3.ex3.train_best_model()
x = EX3.ex3_312237803.predict(model, r'trump_test.tsv')
file = open('312237803.txt', 'w')
file.write(x)
# y = EX3.ex3.predict(trained_model, r'trump_test.tsv')
# print(x)
# print(y)
# print(x==y)

