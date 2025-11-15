import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("comportamento_de_compra.csv")
print("Primeiras 10 linhas:\n", df.head(10))

colunas_para_manter = ["Data", "Cliente", "NomeProduto", "Preco", "Quantidade"]
df = df[colunas_para_manter]

df = df.rename(columns={
    "Data": "data",
    "Cliente": "cliente",
    "NomeProduto": "produto",
    "Preco": "preco",
    "Quantidade": "quantidade"
})

# 3. Converter a coluna Data para datetime
df["data"] = pd.to_datetime(df["data"], errors="coerce")

# 4a. Total de vendas (Preço × Quantidade)
df["total"] = df["preco"] * df["quantidade"]
total_vendas = df["total"].sum()
print("\nTotal de vendas:", total_vendas)

# 4b. Produto mais vendido (por quantidade)
produto_mais_vendido = df.groupby("produto")["quantidade"].sum().idxmax()
print("\nProduto mais vendido:", produto_mais_vendido)

# 4c. Média de compras por cliente
media_por_cliente = df.groupby("cliente")["total"].sum().mean()
print("\nMédia de compras por cliente:", media_por_cliente)

# 5. Gráfico: 5 produtos mais vendidos
top5 = df.groupby("produto")["quantidade"].sum().nlargest(5)

plt.figure(figsize=(8,5))
top5.plot(kind="bar")
plt.title("Top 5 Produtos Mais Vendidos")
plt.xlabel("Produto")
plt.ylabel("Quantidade Vendida")
plt.tight_layout()
plt.show()

# 6. Salvar DataFrame final em um novo arquivo
df.to_csv("vendas_tratadas.csv", index=False)
print("\nArquivo vendas_tratadas.csv salvo com sucesso!")
