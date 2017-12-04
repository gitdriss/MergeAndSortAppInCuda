import pandas as pd

file = pd.read_csv("autos.csv", sep=",")
file2 = file.drop_duplicates(['price']) 
file2 = pd.DataFrame(file2, columns=['price'])
file2.to_csv('price.txt', sep=',', index=False, header=False)

file3 = pd.DataFrame(file, columns=['kilometer'])
file3.to_csv('kilometer.txt', sep=',', index=False, header=False)
