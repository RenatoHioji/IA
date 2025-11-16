import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("comportamento_de_compra_reduzido.csv")
colunas_para_manter = ["Data", "IDCliente", "NomeProduto", "Preco", "Quantidade"]
df = df[colunas_para_manter]

df = df.rename(columns={
    "Data": "data",
    "IDCliente": "cliente",
    "NomeProduto": "produto",
    "Preco": "preco",
    "Quantidade": "quantidade"
})

cesta = df.pivot_table(index="cliente", columns="produto",
                       aggfunc=lambda x: 1, fill_value=0)

cesta = cesta.astype(bool)

frequencias = cesta.sum()
cesta = cesta.loc[:, frequencias >= 2]

frequencia = apriori(cesta, min_support=0.05, use_colnames=True)

regras = association_rules(frequencia, metric="confidence", min_threshold=0.50)

regras_top10 = regras.sort_values(by="lift", ascending=False).head(10)

if regras_top10.empty:
    print("Nenhuma regra encontrada. Diminua o suporte.")
else:
    plt.figure(figsize=(10, 7))

    jitter = 0.002
    suporte_jitter = regras_top10['support'] + np.random.uniform(-jitter, jitter, len(regras_top10))
    confianca_jitter = regras_top10['confidence'] + np.random.uniform(-jitter, jitter, len(regras_top10))

    plt.scatter(
        suporte_jitter,
        confianca_jitter,
        s=regras_top10['lift'] * 500,
        alpha=0.6
    )

    for n, (i, row) in enumerate(regras_top10.iterrows(), start=1):
        plt.annotate(str(n), (suporte_jitter[i], confianca_jitter[i]), fontsize=9, weight="bold")

    plt.xlabel("Suporte")
    plt.ylabel("Confiança")
    plt.title("Top 10 Regras por Lift (Numeradas)")
    plt.tight_layout()
    plt.show()

    print("\n===== LEGENDA DAS REGRAS =====\n")
    for n, (i, row) in enumerate(regras_top10.iterrows(), start=1):
        antecedente = ", ".join(map(str, row['antecedents']))
        consequente = ", ".join(map(str, row['consequents']))
        print(f"{n}. {antecedente}  ->  {consequente} | Lift={row['lift']:.2f}")
