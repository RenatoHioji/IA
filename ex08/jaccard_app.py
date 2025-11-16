import pandas as pd
import itertools
import argparse

def jaccard(a, b):
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="App de Similaridade Jaccard")
    parser.add_argument("--detalhe", nargs=2, help="Comparar dois clientes específicos")
    args = parser.parse_args()

    df = pd.read_csv("compras.csv")

    grupos = df.groupby("cliente")["produto"].apply(set).to_dict()

    if args.detalhe:
        c1, c2 = args.detalhe
        if c1 not in grupos or c2 not in grupos:
            print("Cliente não encontrado.")
            return
        sim = jaccard(grupos[c1], grupos[c2])
        print(f"Similaridade Jaccard entre {c1} e {c2}: {sim:.3f}")
        return

    pares = []
    for c1, c2 in itertools.combinations(grupos.keys(), 2):
        sim = jaccard(grupos[c1], grupos[c2])
        pares.append(((c1, c2), sim))

    pares = sorted(pares, key=lambda x: x[1], reverse=True)

    print("Top 3 pares mais similares:")
    for (c1, c2), score in pares[:3]:
        print(f"{c1} - {c2}: Jaccard = {score:.3f}")

if __name__ == "__main__":
    main()
