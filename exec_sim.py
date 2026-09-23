
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

H_PADRAO = 30                       # horizonte de execucao, em minutos
INICIOS = [10.5, 12.0, 13.5, 15.0]  # inicio das janelas, horario de Brasilia


# ----------------------------------------------------------------- agendadores
def agenda_twap(n: int) -> np.ndarray:
    return np.ones(n) / n


def agenda_pov(vol: np.ndarray) -> np.ndarray:
    return vol / vol.sum()


def agenda_normalizada(s: np.ndarray, k: float) -> np.ndarray:
    """COM VAZAMENTO: a normalizacao usa os pesos de todas as barras da janela,
    inclusive as futuras. Mantida apenas para demonstrar o efeito do erro."""
    p = np.clip(1 + k * s, 0, None)
    return p / p.sum()


def agenda_causal(s: np.ndarray, k: float) -> np.ndarray:
    """Sem vazamento: a cada barra envia uma fracao do inventario remanescente,
    modulada pelo estado ja observado. A ultima barra liquida o que restar."""
    n = len(s)
    rem, w = 1.0, np.zeros(n)
    for i in range(n):
        if i == n - 1:
            w[i] = rem
            break
        fracao = np.clip((1.0 / (n - i)) * (1 + k * s[i]), 0.0, 1.0)
        w[i] = rem * fracao
        rem -= w[i]
    return w


# ----------------------------------------------------------------- simulacao
def simular(barras: pd.DataFrame, horizonte: int = H_PADRAO,
            ks=(-1.0, -0.5, 0.5, 1.0)) -> pd.DataFrame:
    """Uma ordem de compra por (ativo, pregao, janela). Cortes de estado vindos
    apenas dos pregoes anteriores — o primeiro pregao serve so de treino."""
    b = barras.dropna(subset=["mid", "sp", "vol", "ofi_norm"]).copy()
    t = pd.to_datetime(b["jan"])
    b["hora"] = t.dt.hour + t.dt.minute / 60
    datas = sorted(b["data"].unique())

    linhas = []
    for i, data in enumerate(datas):
        if i == 0:
            continue
        lo, hi = b[b["data"].isin(datas[:i])]["ofi_norm"].quantile([1 / 3, 2 / 3])
        for ativo, g in b[b["data"] == data].groupby("ativo"):
            g = g.sort_values("jan").reset_index(drop=True)
            for t0 in INICIOS:
                w = g[(g["hora"] >= t0) & (g["hora"] < t0 + horizonte / 60)]
                if len(w) < horizonte * 0.8 or w["vol"].sum() <= 0:
                    continue
                mid = w["mid"].to_numpy()
                meio_spread = (w["sp"].to_numpy() / 2) * mid
                vol = w["vol"].to_numpy()
                chegada, n = mid[0], len(w)

                ofi = w["ofi_norm"].to_numpy()
                estado = np.where(ofi >= hi, 1, np.where(ofi <= lo, -1, 0))
                s = np.concatenate([[0], estado[:-1]])   # decide em i com o estado de i-1

                politicas = {"TWAP": agenda_twap(n), "POV": agenda_pov(vol)}
                for k in ks:
                    politicas[f"causal k={k:+.1f}"] = agenda_causal(s, k)
                    politicas[f"vazamento k={k:+.1f}"] = agenda_normalizada(s, k)

                for nome, p in politicas.items():
                    preco = float((p * (mid + meio_spread)).sum())
                    linhas.append({"data": str(data), "ativo": ativo, "janela": t0,
                                   "politica": nome,
                                   "is_bps": (preco / chegada - 1) * 1e4})
    return pd.DataFrame(linhas)


def comparar(res: pd.DataFrame, referencia: str = "TWAP") -> pd.DataFrame:
    piv = res.pivot_table(index=["data", "ativo", "janela"],
                          columns="politica", values="is_bps")
    out = []
    for c in piv.columns:
        if c == referencia:
            continue
        d = (piv[referencia] - piv[c]).dropna()
        out.append({"politica": c, "ganho_bps": d.mean(),
                    "t": d.mean() / (d.std() / np.sqrt(len(d))),
                    "vitorias": (d > 0).mean(), "n": len(d)})
    return pd.DataFrame(out).sort_values("ganho_bps", ascending=False)


def selecao_fora_da_amostra(res: pd.DataFrame, prefixo: str) -> dict:
    """Escolhe k nos dois primeiros pregoes de teste e avalia nos dois seguintes.
    Sem esta etapa, o melhor k e escolhido sobre os mesmos dados que o avaliam."""
    piv = res.pivot_table(index=["data", "ativo", "janela"],
                          columns="politica", values="is_bps")
    datas = sorted(piv.index.get_level_values("data").unique())
    sel, ava = piv.loc[datas[:2]], piv.loc[datas[2:]]
    cand = [c for c in piv.columns if c.startswith(prefixo)]
    melhor = max(cand, key=lambda c: (sel["TWAP"] - sel[c]).mean())
    d = (ava["TWAP"] - ava[melhor]).dropna()
    return {"selecionada": melhor, "ganho_bps": d.mean(),
            "t": d.mean() / (d.std() / np.sqrt(len(d))), "n": len(d)}


def main() -> None:
    p = argparse.ArgumentParser(description="Baselines de execucao com sinal de fluxo")
    p.add_argument("--barras", default="barras_mid.pkl",
                   help="pickle com barras de 1min: ativo, data, jan, mid, sp, vol, ofi_norm")
    p.add_argument("--horizonte", type=int, default=H_PADRAO)
    p.add_argument("--out")
    a = p.parse_args()

    barras = pd.read_pickle(a.barras)
    res = simular(barras, horizonte=a.horizonte)
    pd.set_option("display.width", 120, "display.float_format", lambda v: f"{v:,.3f}")

    n_ordens = res.groupby(["data", "ativo", "janela"]).ngroups
    print(f"\nordens simuladas: {n_ordens:,} | ativos: {res.ativo.nunique()}")
    print("\n--- ganho sobre TWAP (bps) ---")
    print(comparar(res).to_string(index=False))
    print("\n--- selecao fora da amostra ---")
    for prefixo in ("causal", "vazamento"):
        print(f"  {prefixo}: {selecao_fora_da_amostra(res, prefixo)}")
    print("\nO sinal do melhor k difere entre as duas familias: e o efeito do vazamento.")

    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
