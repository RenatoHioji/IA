import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vendas_tratadas.csv")

resumo = df.describe()
print("Resumo estatístico:\n", resumo)

plt.figure(figsize=(7,4))
plt.hist(df["preco"])
plt.title("Histograma dos Preços")
plt.xlabel("Preço")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.boxplot(df["quantidade"])
plt.title("Boxplot das Quantidades")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

vendas_produtos = df.groupby("produto")["quantidade"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
vendas_produtos.plot(kind="bar")
plt.title("Pareto - Produtos Mais Vendidos")
plt.xlabel("Produto")
plt.ylabel("Quantidade Vendida")
plt.tight_layout()
plt.show()

media = df["preco"].mean()
mediana = df["preco"].median()
moda = df["preco"].mode()[0]
desvio = df["preco"].std()

print("\nMédia do preço:", media)
print("Mediana do preço:", mediana)
print("Moda do preço:", moda)
print("Desvio padrão do preço:", desvio)
