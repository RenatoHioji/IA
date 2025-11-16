import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("comportamento_de_compra.csv")
print("Primeiras 10 linhas:\n", df.head(10))

colunas_para_manter = ["Data", "IDCliente", "NomeProduto", "Preco", "Quantidade"]
df = df[colunas_para_manter]

df = df.rename(columns={
    "Data": "data",
    "IDCliente": "cliente",
    "NomeProduto": "produto",
    "Preco": "preco",
    "Quantidade": "quantidade"
})

df["data"] = pd.to_datetime(df["data"], errors="coerce")

df["total"] = df["preco"] * df["quantidade"]
total_vendas = df["total"].sum()
print("\nTotal de vendas:", total_vendas)

produto_mais_vendido = df.groupby("produto")["quantidade"].sum().idxmax()
print("\nProduto mais vendido:", produto_mais_vendido)

media_por_cliente = df.groupby("cliente")["total"].sum().mean()
print("\nMédia de compras por cliente:", media_por_cliente)

top5 = df.groupby("produto")["quantidade"].sum().nlargest(5)

plt.figure(figsize=(8,5))
top5.plot(kind="bar")
plt.title("Top 5 Produtos Mais Vendidos")
plt.xlabel("Produto")
plt.ylabel("Quantidade Vendida")
plt.tight_layout()
plt.show()

df.to_csv("vendas_tratadas.csv", index=False)
print("\nArquivo vendas_tratadas.csv salvo com sucesso!")
