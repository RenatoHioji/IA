import pandas as pd
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

produtos = [
    "Camiseta", "Calça Jeans", "Jaqueta", "Tênis", "Bolsa", "Óculos de Sol",
    "Relógio", "Carteira", "Brinco", "Sandália", "Chinelo", "Boné"
]

def gerar_transacao():
    principal = random.choice([
        ["Camiseta", "Calça Jeans", "Tênis", "Boné", "Cinto", "Meias"],
        ["Vestido", "Sandália", "Bolsa", "Brinco"],
        ["Jaqueta", "Calça Jeans", "Tênis"],
        ["Shorts", "Camiseta", "Sandália"],
        ["Tênis", "Meias", "Boné", "Camiseta"]
    ])

    transacao = random.sample(principal, random.randint(2, len(principal)))

    acessorios = list(set(produtos) - set(principal))
    if random.random() < 0.4:
        transacao += random.sample(acessorios, k=1)

    return list(set(transacao))

random.seed(42)
num_transacoes = 50
transacoes = [gerar_transacao() for _ in range(num_transacoes)]

itens = sorted(set(item for trans in transacoes for item in trans))
cols = list(itens)

df_trans = pd.DataFrame(0, index=range(num_transacoes), columns=cols)

for i, trans in enumerate(transacoes):
    for item in trans:
        df_trans.loc[i, item] = 1

csv_path = "regras_mercado.csv"
df_trans.to_csv(csv_path, index=False)
print(f"✔️ Arquivo CSV salvo com sucesso: {csv_path}")

df = pd.read_csv(csv_path)

transacoes_processadas = df.apply(lambda row: [item for item in row.index if row[item] == 1], axis=1)

encoder = TransactionEncoder()
te_array = encoder.fit(transacoes_processadas).transform(transacoes_processadas)
df_encoded = pd.DataFrame(te_array, columns=encoder.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.03, use_colnames=True)

print("\n📌 Itens frequentes:")
print(frequent_itemsets)

# Regras de associação
regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
melhor_regra = regras.loc[regras['lift'].idxmax()]


if not regras.empty:
    for _, row in regras.iterrows():
        print(f"Regra: {set(row['antecedents'])} -> {set(row['consequents'])}")
        print(f" - Suporte: {row['support']}")
        print(f" - Confiança: {row['confidence']}")
        print(f" - Lift: {row['lift']}\n")
else:
    print("⚠️ Nenhuma regra encontrada com os parâmetros definidos.")

print("\n📌 Regras de associação (com suporte, confiança e lift):\n")
print(f"Melhor regra {melhor_regra}")
