import pandas as pd
import itertools

df = pd.read_csv("compras.csv")

clientes = df.groupby("cliente")["produto"].apply(set)

def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union != 0 else 0

similaridades = []

for c1, c2 in itertools.combinations(clientes.index, 2):
    sim = jaccard(clientes[c1], clientes[c2])
    similaridades.append((c1, c2, sim))

similaridades = sorted(similaridades, key=lambda x: x[2], reverse=True)

def analisar_cliente(cliente_escolhido, top_n=3):
    ranking = []

    for outro in clientes.index:
        if outro == cliente_escolhido:
            continue
        sim = jaccard(clientes[cliente_escolhido], clientes[outro])
        ranking.append((outro, sim))

    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)

    print(f"\nClientes mais semelhantes a {cliente_escolhido}:")
    for r in ranking[:top_n]:
        print(f"{r[0]} -> Similaridade {r[1]:.2f}")

    semelhantes = [r[0] for r in ranking[:top_n]]

    produtos_semelhantes = []
    for c in semelhantes:
        produtos_semelhantes.extend(list(clientes[c]))

    freq = pd.Series(produtos_semelhantes).value_counts()

    recomendados = freq.index.difference(clientes[cliente_escolhido])

    print(f"\nProdutos mais frequentes entre os semelhantes de {cliente_escolhido}:")
    print(freq)

    print(f"\nRecomendações para {cliente_escolhido}:")
    print(list(recomendados))


analisar_cliente("A")
