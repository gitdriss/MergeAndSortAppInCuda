import pandas as pd

file = pd.read_csv("../data/autos.csv", sep=";")
file2 = file.drop_duplicates(['price'])
file2 = pd.DataFrame(file2, columns=['price'])
file2.to_csv('../data/price.txt', sep=';', index=False, header=False)

file3 = file.drop_duplicates(['kilometer']) 
file3 = pd.DataFrame(file3, columns=['kilometer'])
file3.to_csv('../data/kilometer.txt', sep=';', index=False, header=False)
