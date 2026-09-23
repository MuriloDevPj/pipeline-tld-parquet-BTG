"""
Pipeline BTG-TLD-A26: ticks -> order flow -> Markov -> regras -> validacao.
Uso:
    python btg_tld_pipeline.py --demo
    python btg_tld_pipeline.py --path dados/ --freq 1min --out saida
    python btg_tld_pipeline.py --path dados/ --freq 1min --pregao 10:05-16:50 --out saida
    python btg_tld_pipeline.py --path dados/ --tune --out saida
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

# BTG-TLD-A26 nao tem coluna "timestamp": o tempo do evento e md_entry_datetime.
CANDIDATOS_TS = ["timestamp", "md_entry_datetime", "network_received_time", "sending_time"]
COLS_NUCLEO = ["md_entry_px", "md_entry_size", "aggressor", "tick_direction"]
COLS_EXTRA = ["symbol", "seq_num", "rpt_seq", "trade_id", "md_update_action",
              "trade_condition", "top_px_bid", "top_px_offer"]

# Convencao do aggressor: NAO e universal. Rode --diag para inferir do proprio dado
# e sobrescreva com --mapa "3:1,4:-1". Sinal trocado inverte o backtest sem dar erro.
MAPA_AGGRESSOR = {
    1: 1, "1": 1, "B": 1, "BUY": 1, "BID": 1, "COMPRA": 1, "COMPRADOR": 1,
    2: -1, "2": -1, "S": -1, "SELL": -1, "ASK": -1, "VENDA": -1, "VENDEDOR": -1,
    0: 0, "0": 0, "N": 0, "NONE": 0, "": 0,
}
TZ_ORIGEM: str | None = "UTC"                 # dataset vem em UTC
TZ_DESTINO: str | None = "America/Sao_Paulo"  # analise em horario de pregao


def resolver_colunas(disponiveis) -> tuple[str, list[str]]:
    """Descobre a coluna de tempo e monta a lista minima a ler do parquet."""
    disp = set(disponiveis)
    ts = next((c for c in CANDIDATOS_TS if c in disp), None)
    if ts is None:
        raise KeyError(f"nenhuma coluna de tempo encontrada entre {CANDIDATOS_TS}. Schema: {sorted(disp)}")
    faltando = [c for c in COLS_NUCLEO if c not in disp]
    if faltando:
        raise KeyError(f"colunas essenciais ausentes: {faltando}")
    return ts, [ts] + COLS_NUCLEO + [c for c in COLS_EXTRA if c in disp]
ROTULOS = {0: "forte_venda", 1: "neutro", 2: "forte_compra"}


# ============================================================================
# PASSO 1 - Ingestao
# ============================================================================
def preparar(df: pd.DataFrame, ts_col: str | None = None,
             tz_origem: str | None = None, tz_destino: str | None = None) -> pd.DataFrame:
    """Tipagem enxuta, fuso, ordenacao cronologica desempatada e sinal de agressao."""
    df = df.copy()
    if "timestamp" not in df.columns:
        ts_col = ts_col or next((c for c in CANDIDATOS_TS if c in df.columns), None)
        if ts_col is None:
            raise KeyError(f"coluna de tempo nao encontrada entre {CANDIDATOS_TS}")
        df = df.rename(columns={ts_col: "timestamp"})

    t = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    tz_o = TZ_ORIGEM if tz_origem is None else tz_origem
    tz_d = TZ_DESTINO if tz_destino is None else tz_destino
    if tz_o and tz_d:                       # UTC no arquivo -> horario de Sao Paulo
        t = (t.dt.tz_localize(tz_o) if t.dt.tz is None else t.dt.tz_convert(tz_o))
        t = t.dt.tz_convert(tz_d).dt.tz_localize(None)
    df["timestamp"] = t

    df["md_entry_px"] = pd.to_numeric(df["md_entry_px"], errors="coerce").astype("float64")
    df["md_entry_size"] = pd.to_numeric(df["md_entry_size"], errors="coerce").astype("float64")

    agg = df["aggressor"]
    if agg.dtype == object or isinstance(agg.dtype, pd.CategoricalDtype):
        agg = agg.astype(str).str.strip().str.upper()
    df["sinal"] = agg.map(MAPA_AGGRESSOR).fillna(0).astype("int8")
    df["tick_direction"] = pd.to_numeric(df["tick_direction"], errors="coerce").astype("Int8")

    df = df.dropna(subset=["timestamp", "md_entry_px", "md_entry_size"])
    df = df[(df["md_entry_px"] > 0) & (df["md_entry_size"] > 0)]

    # desempate: varios eventos compartilham o mesmo ms; seq_num/rpt_seq dao a ordem real
    chaves = ["timestamp"] + [c for c in ("seq_num", "rpt_seq") if c in df.columns]
    df = df.sort_values(chaves, kind="mergesort").reset_index(drop=True)

    df["data"] = df["timestamp"].dt.date
    if "ativo" not in df.columns:
        df["ativo"] = df["symbol"] if "symbol" in df.columns else "UNICO"
    df["sessao"] = df["ativo"].astype(str) + "|" + df["data"].astype(str)
    return df


def carregar_ticks(path: str, colunas: list[str] | None = None) -> pd.DataFrame:
    """Le tudo de uma vez. Use so se a base couber na RAM; caso contrario, streaming."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    ts, cols = resolver_colunas(dataset.schema.names)
    return preparar(dataset.to_table(columns=colunas or cols).to_pandas(), ts_col=ts)


def barras_streaming(path: str, freq: str | None = "1min", ticks: int | None = None,
                     colunas: list[str] | None = None, batch_size: int = 500_000,
                     pregao: tuple[str, str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Processa a base INTEIRA sem carrega-la toda na memoria.

    Le em lotes do Arrow, mantem em RAM apenas os ticks da(s) sessao(oes) em aberto
    e descarrega cada sessao em barras assim que ela termina. As barras sao ~4 ordens
    de grandeza menores que os ticks, entao o resultado final cabe folgado.
    """
    import pyarrow.dataset as ds

    colunas = colunas or COLS
    dataset = ds.dataset(path, format="parquet")
    faltando = [c for c in colunas if c not in set(dataset.schema.names)]
    if faltando:
        raise KeyError(f"colunas ausentes: {faltando}. Schema: {sorted(dataset.schema.names)}")

    buffers: dict[object, list[pd.DataFrame]] = {}
    saida: list[pd.DataFrame] = []
    total = 0

    def flush(sessoes) -> None:
        for s in sessoes:
            chunk = preparar(pd.concat(buffers.pop(s), ignore_index=True))
            if pregao:
                chunk = filtrar_pregao(chunk, *pregao)
            if not chunk.empty:
                saida.append(construir_barras(chunk, freq=freq, ticks=ticks))

    for lote in dataset.scanner(columns=colunas, batch_size=batch_size).to_batches():
        d = preparar(lote.to_pandas())
        if d.empty:
            continue
        total += len(d)
        vistas = set()
        for s, g in d.groupby("sessao", sort=True):
            buffers.setdefault(s, []).append(g)
            vistas.add(s)
        # sessao que nao apareceu neste lote esta encerrada -> vira barras e sai da RAM
        flush([s for s in list(buffers) if s not in vistas])
        if verbose:
            print(f"\r  ticks lidos: {total:,} | sessoes fechadas: {len(saida):,}", end="", file=sys.stderr)

    flush(list(buffers))
    if verbose:
        print(f"\r  ticks lidos: {total:,} | sessoes: {len(saida):,}", file=sys.stderr)
    if not saida:
        raise ValueError("nenhuma barra gerada - verifique o filtro de pregao e os dados")

    barras = pd.concat(saida, ignore_index=True).sort_values(["sessao", "janela"]).reset_index(drop=True)
    return _derivar(barras)


def filtrar_pregao(df: pd.DataFrame, inicio: str = "10:05", fim: str = "16:50") -> pd.DataFrame:
    """Corta leilao de abertura/fechamento e after-market, onde a microestrutura muda de regime."""
    h = df["timestamp"].dt.time
    return df[(h >= pd.Timestamp(inicio).time()) & (h <= pd.Timestamp(fim).time())]



# ============================================================================
# Base multiativo: um arquivo por (ticker, pregao)
# ============================================================================
import glob as _glob
import re as _re

PADRAO_NOME = _re.compile(r"(?P<ativo>[A-Z0-9]+)_(?P<data>\d{4}-\d{2}-\d{2})\.parquet$", _re.I)


def descobrir_arquivos(path: str, ticker: str | None = None, datas: tuple[str, str] | None = None) -> pd.DataFrame:
    """Mapeia TICKER_AAAA-MM-DD.parquet. Aceita arquivo unico, pasta ou glob."""
    if os.path.isdir(path):
        arquivos = sorted(_glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    else:
        arquivos = sorted(_glob.glob(path)) or [path]

    linhas = []
    for f in arquivos:
        m = PADRAO_NOME.search(os.path.basename(f))
        linhas.append({"arquivo": f,
                       "ativo": m.group("ativo").upper() if m else os.path.basename(f).split(".")[0],
                       "data": m.group("data") if m else None})
    cat = pd.DataFrame(linhas)
    if ticker:
        alvos = {t.strip().upper() for t in ticker.split(",")}
        cat = cat[cat["ativo"].isin(alvos)]
    if datas:
        cat = cat[(cat["data"] >= datas[0]) & (cat["data"] <= datas[1])]
    if cat.empty:
        raise ValueError(f"nenhum parquet encontrado em {path} com os filtros aplicados")
    return cat.sort_values(["data", "ativo"]).reset_index(drop=True)


def inspecionar(path: str, n: int = 5) -> None:
    """Mostre o schema ANTES de rodar qualquer modelo: nomes de coluna e convencao
    do aggressor mudam entre provedores e um sinal invertido nao gera erro nenhum."""
    import pyarrow.parquet as pq

    cat = descobrir_arquivos(path)
    f = cat["arquivo"].iloc[0]
    pf = pq.ParquetFile(f)
    print(f"arquivos: {len(cat):,} | ativos: {cat['ativo'].nunique()} | datas: {sorted(cat['data'].dropna().unique())}")
    print(f"\namostra: {f}")
    print(f"linhas: {pf.metadata.num_rows:,} | tamanho: {os.path.getsize(f)/1e6:,.1f} MB")
    print("\n--- schema ---")
    for c in pf.schema_arrow:
        print(f"  {c.name}: {c.type}")
    df = pf.read().to_pandas().head(2000)
    print("\n--- primeiras linhas ---")
    print(df.head(n).to_string())
    for col in ("aggressor", "tick_direction", "entry_type", "md_entry_type"):
        if col in df.columns:
            print(f"\n--- distribuicao de {col} ---")
            print(df[col].value_counts(dropna=False).head(10).to_string())
    ts_col, cols = resolver_colunas(df.columns)
    print(f"\ncoluna de tempo detectada: {ts_col} | colunas lidas: {cols}")
    bruto = pd.to_datetime(pf.read(columns=[ts_col]).to_pandas()[ts_col], errors="coerce", format="mixed")
    print(f"faixa horaria NO ARQUIVO: {bruto.min()} a {bruto.max()}")
    print(f"apos conversao {TZ_ORIGEM}->{TZ_DESTINO}: "
          f"{bruto.dt.tz_localize(TZ_ORIGEM).dt.tz_convert(TZ_DESTINO).min()} a "
          f"{bruto.dt.tz_localize(TZ_ORIGEM).dt.tz_convert(TZ_DESTINO).max()}")


# ----------------------------------------------------------------------------
# Leitura de parquet: pyarrow quando disponivel, leitor proprio como reserva
# ----------------------------------------------------------------------------
def _tem_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def ler_parquet(arquivo: str, colunas: list[str] | None = None) -> pd.DataFrame:
    if _tem_pyarrow():
        import pyarrow.parquet as pq
        return pq.read_table(arquivo, columns=colunas).to_pandas()
    from miniparquet import ParquetFile
    pf = ParquetFile(arquivo)
    d = {c: pf.ler_coluna(c) for c in (colunas or pf.nomes)}
    df = pd.DataFrame(d)
    for c in df.columns:                      # BYTE_ARRAY vem como bytes
        if df[c].dtype == object and len(df) and isinstance(df[c].dropna().iloc[:1].squeeze(), bytes):
            df[c] = df[c].str.decode("utf-8")
    return df


def nomes_colunas(arquivo: str) -> list[str]:
    if _tem_pyarrow():
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(arquivo).schema_arrow.names)
    from miniparquet import ParquetFile
    return ParquetFile(arquivo).nomes


def diagnosticar(path: str, ticker: str | None = None, n_arquivos: int = 3) -> pd.DataFrame:
    """Infere a convencao do campo `aggressor` a partir do proprio dado.

    Logica: agressao compradora consome a oferta e empurra o preco para cima, entao
    deve concentrar upticks (tick_direction 0/1) e variacao de preco positiva no
    proprio evento. O valor com maior fracao de uptick e o lado comprador.
    """
    import pyarrow.parquet as pq

    cat = descobrir_arquivos(path, ticker=ticker).head(n_arquivos)
    partes = []
    for f in cat["arquivo"]:
        pfile = pq.ParquetFile(f)
        ts_col, cols = resolver_colunas(pfile.schema_arrow.names)
        partes.append(pfile.read(columns=cols).to_pandas())
    d = pd.concat(partes, ignore_index=True)

    d["_ts"] = pd.to_datetime(d[ts_col], errors="coerce", format="mixed")
    chaves = ["_ts"] + [c for c in ("seq_num", "rpt_seq") if c in d.columns]
    d = d.sort_values(chaves, kind="mergesort")
    d["_px"] = pd.to_numeric(d["md_entry_px"], errors="coerce")
    d["_td"] = pd.to_numeric(d["tick_direction"], errors="coerce")
    d["_dpx"] = d["_px"].diff()

    g = d.groupby("aggressor", dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(d),
        "frac_uptick": g["_td"].apply(lambda s: s.isin([0, 1]).sum() / max(s.notna().sum(), 1)),
        "frac_downtick": g["_td"].apply(lambda s: s.isin([2, 3]).sum() / max(s.notna().sum(), 1)),
        "dpx_medio": g["_dpx"].mean(),
        "size_mediano": g["md_entry_size"].median(),
        "frac_com_trade_id": g.apply(lambda x: x["trade_id"].notna().mean() if "trade_id" in x else np.nan,
                                     include_groups=False),
    })
    out["veredito"] = np.where(out["frac_uptick"] > 0.55, "COMPRADOR (+1)",
                        np.where(out["frac_downtick"] > 0.55, "VENDEDOR (-1)", "indefinido (0)"))

    print(f"arquivos amostrados: {len(cat)} | linhas: {len(d):,}")
    print("\n--- aggressor x direcao do tick ---")
    print(out.to_string())
    for col in ("md_update_action", "trade_condition"):
        if col in d.columns:
            print(f"\n--- {col} x aggressor ---")
            print(pd.crosstab(d[col], d["aggressor"]).head(8).to_string())
    sug = {int(v): (1 if r["veredito"].startswith("COMPRADOR") else -1 if r["veredito"].startswith("VENDEDOR") else 0)
           for v, r in out.iterrows() if pd.notna(v)}
    print("\n--- mapeamento sugerido ---")
    print("  --mapa \"" + ",".join(f"{k}:{v}" for k, v in sug.items() if v != 0) + "\"")
    print("  (confira se faz sentido antes de usar; empate ou tudo indefinido = o campo"
          " nao e lado de agressao e o OFI precisa vir de outra fonte)")
    return out


def aplicar_mapa(texto: str) -> None:
    """--mapa "3:1,4:-1" sobrescreve a convencao do aggressor."""
    novo = {}
    for par in texto.split(","):
        k, v = par.split(":")
        k, v = k.strip(), int(v)
        novo[k] = v
        try:
            novo[int(k)] = v
        except ValueError:
            pass
    MAPA_AGGRESSOR.clear()
    MAPA_AGGRESSOR.update(novo)


def barras_multiativo(path: str, freq: str | None = "1min", ticks: int | None = None,
                      ticker: str | None = None, datas: tuple[str, str] | None = None,
                      pregao: tuple[str, str] | None = None, colunas: list[str] | None = None,
                      verbose: bool = True) -> pd.DataFrame:
    """Le arquivo por arquivo (cada um = 1 ativo-dia, cabe na RAM), agrega em barras
    e descarta os ticks. Escala para a base inteira sem estourar memoria."""
    cat = descobrir_arquivos(path, ticker=ticker, datas=datas)
    ts_col, cols = resolver_colunas(nomes_colunas(cat["arquivo"].iloc[0]))
    colunas = colunas or cols
    saida, lidos, ignorados = [], 0, []

    for k, row in cat.iterrows():
        try:
            d = ler_parquet(row["arquivo"], colunas)
        except Exception as e:
            ignorados.append((row["arquivo"], str(e)[:80]))
            continue
        d["ativo"] = row["ativo"]
        d = preparar(d, ts_col=ts_col)
        if pregao:
            d = filtrar_pregao(d, *pregao)
        lidos += len(d)
        if not d.empty:
            saida.append(construir_barras(d, freq=freq, ticks=ticks))
        if verbose and (k % 25 == 0 or k == len(cat) - 1):
            print(f"\r  arquivos {k+1}/{len(cat)} | ticks {lidos:,}", end="", file=sys.stderr)

    if verbose:
        print("", file=sys.stderr)
    if ignorados:
        print(f"[aviso] {len(ignorados)} arquivo(s) ignorado(s). Ex: {ignorados[0]}", file=sys.stderr)
    if not saida:
        raise ValueError("nenhuma barra gerada")
    b = pd.concat(saida, ignore_index=True).sort_values(["ativo", "data", "janela"]).reset_index(drop=True)
    return _derivar(b)


def resumo_por_ativo(barras: pd.DataFrame) -> pd.DataFrame:
    """Onde o OFI tem poder preditivo, ativo a ativo. Base para escolher o universo."""
    d = barras.dropna(subset=["retorno_fwd", "ofi_norm"])
    out = d.groupby("ativo").agg(barras=("ativo", "size"), ticks=("n_ticks", "sum"),
                                 sigma=("retorno", "std"), volume=("volume", "mean"))
    out["corr_ofi_retfwd"] = d.groupby("ativo").apply(
        lambda g: g["ofi_norm"].corr(g["retorno_fwd"]), include_groups=False)
    out["ep_h0"] = 1 / np.sqrt(out["barras"].clip(lower=2))
    out["t"] = out["corr_ofi_retfwd"] / out["ep_h0"]
    return out.sort_values("t", ascending=False)


# ============================================================================
# PASSO 2 - Features de order flow
# ============================================================================
def construir_barras(df: pd.DataFrame, freq: str | None = "1min", ticks: int | None = None,
                     derivar: bool = False) -> pd.DataFrame:
    if (freq is None) == (ticks is None):
        raise ValueError("defina freq OU ticks, nao ambos")

    d = df.copy()
    d["janela"] = (d.groupby("sessao").cumcount() // ticks) if ticks else d["timestamp"].dt.floor(freq)
    d["_pxvol"] = d["md_entry_px"] * d["md_entry_size"]
    d["_vol_assinado"] = d["md_entry_size"] * d["sinal"]
    d["_vol_compra"] = d["md_entry_size"].where(d["sinal"] > 0, 0.0)
    d["_vol_venda"] = d["md_entry_size"].where(d["sinal"] < 0, 0.0)
    d["_up"] = d["tick_direction"].isin([0, 1]).astype("int16")
    d["_down"] = d["tick_direction"].isin([2, 3]).astype("int16")

    b = d.groupby(["sessao", "janela"], sort=True).agg(
        ativo=("ativo", "first"), data=("data", "first"),
        inicio=("timestamp", "first"), fim=("timestamp", "last"),
        n_ticks=("md_entry_px", "size"),
        abertura=("md_entry_px", "first"), fechamento=("md_entry_px", "last"),
        maxima=("md_entry_px", "max"), minima=("md_entry_px", "min"),
        _pxvol=("_pxvol", "sum"), volume=("md_entry_size", "sum"),
        vol_compra=("_vol_compra", "sum"), vol_venda=("_vol_venda", "sum"),
        ofi=("_vol_assinado", "sum"),
        lote_max=("md_entry_size", "max"), lote_mediano=("md_entry_size", "median"),
        _u=("_up", "sum"), _d=("_down", "sum"),
    ).reset_index()

    b["vwap"] = b["_pxvol"] / b["volume"]
    b["saldo_tick"] = b["_u"] - b["_d"]
    b = b.drop(columns=["_pxvol", "_u", "_d"])
    b = b[b["n_ticks"] > 0].reset_index(drop=True)
    return _derivar(b) if derivar else b


def _derivar(b: pd.DataFrame) -> pd.DataFrame:
    """Colunas que dependem da serie completa - aplicadas so depois de juntar as sessoes."""
    b = b.copy()
    b["ofi_norm"] = b["ofi"] / b["volume"].replace(0, np.nan)
    b["retorno"] = np.log(b["fechamento"]).groupby(b["sessao"]).diff()
    b["retorno_fwd"] = b.groupby("sessao")["retorno"].shift(-1)
    # sigma intradiaria: EWMA dos retornos da propria sessao, defasada 1 barra (sem look-ahead)
    b["sigma"] = (
        b.groupby("sessao")["retorno"]
        .transform(lambda s: s.ewm(span=30, min_periods=8).std().shift(1))
    )
    b["sigma"] = b["sigma"].fillna(b.groupby("sessao")["sigma"].transform("median")).fillna(b["sigma"].median())
    b["hora"] = pd.to_datetime(b["inicio"]).dt.hour
    return b


def perfil_intradiario(b: pd.DataFrame, freq_bucket: str = "30min") -> pd.DataFrame:
    """Robustez por horario: liquidez, volatilidade e poder preditivo do OFI faixa a faixa."""
    t = pd.to_datetime(b["inicio"])
    bucket = t.dt.floor(freq_bucket).dt.strftime("%H:%M")
    d = b.assign(faixa=bucket).dropna(subset=["retorno_fwd", "ofi_norm"])
    out = d.groupby("faixa").agg(
        barras=("faixa", "size"),
        ticks_medios=("n_ticks", "mean"),
        volume_medio=("volume", "mean"),
        sigma=("retorno", "std"),
        ofi_medio=("ofi_norm", "mean"),
        corr_ofi_retfwd=("ofi_norm", lambda s: s.corr(d.loc[s.index, "retorno_fwd"])),
    )
    # erro-padrao da correlacao sob H0: ~1/sqrt(n). |corr| < 2*ep = ruido.
    out["ep_h0"] = 1 / np.sqrt(out["barras"].clip(lower=2))
    out["significante"] = out["corr_ofi_retfwd"].abs() > 2 * out["ep_h0"]
    return out


# ============================================================================
# PASSO 3 - Cadeia de Markov
# ============================================================================
def discretizar(b: pd.DataFrame, coluna: str = "ofi_norm", q: tuple[float, float] = (0.33, 0.67),
                cortes: tuple[float, float] | None = None) -> tuple[pd.Series, tuple[float, float]]:
    x = b[coluna]
    if cortes is None:
        cortes = (float(x.quantile(q[0])), float(x.quantile(q[1])))
    lo, hi = cortes
    e = pd.Series(1, index=b.index, dtype="int8")
    e[x <= lo] = 0
    e[x >= hi] = 2
    return e, (lo, hi)


def matriz_transicao(estados: pd.Series, sessoes: pd.Series, n: int = 3, alpha: float = 1.0) -> pd.DataFrame:
    atual, prox = estados.to_numpy(), estados.shift(-1).to_numpy()
    mesma = sessoes.to_numpy() == sessoes.shift(-1).to_numpy()
    ok = mesma & ~pd.isna(prox)
    c = np.full((n, n), alpha)
    np.add.at(c, (atual[ok].astype(int), prox[ok].astype(int)), 1.0)
    idx = [ROTULOS[i] for i in range(n)]
    return pd.DataFrame(c / c.sum(axis=1, keepdims=True), index=idx, columns=idx)


def teste_markov(estados: pd.Series, sessoes: pd.Series, n: int = 3) -> dict:
    """Qui-quadrado de independencia: a cadeia carrega informacao ou e i.i.d.?"""
    atual, prox = estados.to_numpy(), estados.shift(-1).to_numpy()
    ok = (sessoes.to_numpy() == sessoes.shift(-1).to_numpy()) & ~pd.isna(prox)
    obs = np.zeros((n, n))
    np.add.at(obs, (atual[ok].astype(int), prox[ok].astype(int)), 1.0)
    tot = obs.sum()
    esp = np.outer(obs.sum(1), obs.sum(0)) / tot
    qui = float(((obs - esp) ** 2 / np.maximum(esp, 1e-9)).sum())
    gl = (n - 1) ** 2
    critico_5pct = {1: 3.84, 2: 5.99, 4: 9.49, 9: 16.92}.get(gl, np.nan)
    return {"qui2": qui, "gl": gl, "critico_5pct": critico_5pct,
            "rejeita_independencia": qui > critico_5pct, "n_transicoes": int(tot)}


def estatisticas_por_estado(b: pd.DataFrame, estados: pd.Series) -> pd.DataFrame:
    """Em barras de 1min boa parte dos retornos e exatamente zero (tick de R$0,01).

    Por isso o sinal usa p_alta_cond = P(subir | houve movimento); a versao
    incondicional faria todo estado parecer baixista e disparar venda em tudo.
    """
    d = pd.DataFrame({"estado": estados.map(ROTULOS), "r": b["retorno_fwd"]}).dropna()
    g = d.groupby("estado")["r"]
    out = pd.DataFrame({
        "n": g.size(),
        "p_alta": g.apply(lambda s: (s > 0).mean()),
        "p_baixa": g.apply(lambda s: (s < 0).mean()),
        "p_zero": g.apply(lambda s: (s == 0).mean()),
        "retorno_medio": g.mean(),
        "desvio": g.std(),
    })
    mov = out["p_alta"] + out["p_baixa"]
    out["p_alta_cond"] = out["p_alta"] / mov.replace(0, np.nan)
    out["n_mov"] = (out["n"] * mov).round()
    out["ep_p"] = np.sqrt(0.25 / out["n_mov"].clip(lower=1))
    out["z_vs_50"] = (out["p_alta_cond"] - 0.5) / out["ep_p"]
    return out


def distribuicao_estacionaria(P: pd.DataFrame) -> pd.Series:
    vals, vecs = np.linalg.eig(P.to_numpy().T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return pd.Series(v / v.sum(), index=P.index)


# ============================================================================
# PASSO 4 - Regras, risco e execucao
# ============================================================================
def gerar_sinais(b: pd.DataFrame, estados: pd.Series, stats: pd.DataFrame, limiar: float = 0.55,
                 q_lote: float = 0.99, min_ticks: int = 5, teto_lote: float | None = None,
                 coerencia: bool = False) -> pd.DataFrame:
    """Probabilidade do estado + filtros deterministicos de execucao."""
    d = b.copy()
    d["estado"] = estados.map(ROTULOS)
    d["p_alta"] = d["estado"].map(stats["p_alta_cond"]).fillna(0.5)

    teto = d["lote_max"].quantile(q_lote) if teto_lote is None else teto_lote
    apto = (
        (d["lote_max"] <= teto)                                   # lote anomalo / negocio de bloco
        & (d["n_ticks"] >= min_ticks)                             # barra rala
        & d["sigma"].notna() & (d["sigma"] > 0)                   # risco dimensionavel
    )
    if coerencia:
        # so vale se a relacao OFI->retorno for de CONTINUACAO; com reversao este
        # filtro elimina exatamente os trades certos. Desligado por padrao.
        apto &= np.sign(d["ofi"]) == np.sign(d["p_alta"] - 0.5)
    d["apto"] = apto
    d["sinal_trade"] = 0
    d.loc[d["apto"] & (d["p_alta"] >= limiar), "sinal_trade"] = 1
    d.loc[d["apto"] & (d["p_alta"] <= 1 - limiar), "sinal_trade"] = -1
    return d


def backtest(d: pd.DataFrame, k_stop: float = 1.5, k_alvo: float = 2.5, max_hold: int = 5,
             custo: float = 0.0002, perda_diaria: float = 0.01, trailing: bool = False) -> pd.DataFrame:
    """Sequencial e sem sobreposicao: uma posicao por vez.

    Stop e alvo em multiplos da sigma intradiaria (nao em % fixo), entao a mesma regra
    vale na abertura volatil e no meio do pregao morto. Circuit breaker encerra a sessao
    quando a perda acumulada do dia passa de `perda_diaria`.
    """
    px, hi, lo = d["fechamento"].to_numpy(), d["maxima"].to_numpy(), d["minima"].to_numpy()
    sig, lado_arr = d["sigma"].to_numpy(), d["sinal_trade"].to_numpy()
    sessao = d["sessao"].to_numpy()

    trades, i, pnl_dia, dia_atual = [], 0, 0.0, None
    while i < len(d):
        if sessao[i] != dia_atual:
            dia_atual, pnl_dia = sessao[i], 0.0
        lado = int(lado_arr[i])
        if lado == 0 or pnl_dia <= -perda_diaria or not np.isfinite(sig[i]) or sig[i] <= 0:
            i += 1
            continue

        stop, alvo = k_stop * sig[i], k_alvo * sig[i]
        entrada, pico, r, motivo, j = px[i], 0.0, None, "tempo", i
        for j in range(i + 1, min(i + 1 + max_hold, len(d))):
            if sessao[j] != dia_atual:
                j -= 1
                break
            r_fav = lado * (hi[j] / entrada - 1) if lado > 0 else lado * (lo[j] / entrada - 1)
            r_con = lado * (lo[j] / entrada - 1) if lado > 0 else lado * (hi[j] / entrada - 1)
            gatilho = -stop if not trailing else max(-stop, pico - stop)
            if r_con <= gatilho:                    # stop tem prioridade (hipotese conservadora)
                r, motivo = gatilho, "stop"
                break
            if r_fav >= alvo:
                r, motivo = alvo, "alvo"
                break
            pico = max(pico, r_fav)
        if r is None:
            r = lado * (px[j] / entrada - 1)
        r -= custo
        pnl_dia += r
        trades.append({"i": i, "saida": j, "sessao": dia_atual, "lado": lado, "hora": d["hora"].iloc[i],
                       "p_alta": d["p_alta"].iloc[i], "sigma": sig[i], "retorno": r, "motivo": motivo})
        i = j + 1                                   # sem sobreposicao de posicao
    return pd.DataFrame(trades)


def metricas(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"trades": 0, "taxa_acerto": np.nan, "profit_factor": np.nan,
                "retorno_total": 0.0, "retorno_medio": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    r = t["retorno"]
    ganhos, perdas = r[r > 0].sum(), -r[r < 0].sum()
    eq = r.cumsum()
    return {"trades": len(r), "taxa_acerto": float((r > 0).mean()),
            "profit_factor": float(ganhos / perdas) if perdas > 0 else np.inf,
            "retorno_total": float(r.sum()), "retorno_medio": float(r.mean()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() else np.nan,
            "max_dd": float((eq - eq.cummax()).min())}


# ============================================================================
# Validacao: walk-forward + permutacao
# ============================================================================
def walk_forward(barras: pd.DataFrame, n_blocos: int = 5, q: tuple[float, float] = (0.33, 0.67),
                 limiar: float = 0.55, k_stop: float = 1.5, k_alvo: float = 2.5,
                 max_hold: int = 5, custo: float = 0.0002, embaralhar_estados: bool = False,
                 rng: np.random.Generator | None = None, chave: str = "data") -> pd.DataFrame:
    """Treina em [0..k], testa em [k+1]. Cortes, stats e teto de lote vem SO do treino.

    chave="data": painel - treina em todos os ativos ate o dia k, testa no dia k+1.
    Com 5 pregoes por ticker e a unica forma de ter amostra suficiente por bloco.
    chave="ativo": testa generalizacao para papeis nao vistos no treino.
    """
    if chave not in barras.columns:
        chave = "sessao"
    grupos = pd.Index(barras[chave].dropna().unique()).sort_values()
    n_blocos = min(n_blocos, len(grupos))
    if n_blocos >= 2:
        blocos = np.array_split(np.asarray(grupos), n_blocos)
        fatias = [barras[barras[chave].isin(bl)] for bl in blocos]
    else:
        fatias = np.array_split(barras, 5)

    rng = rng or np.random.default_rng(0)
    todos = []
    for k in range(1, len(fatias)):
        treino = pd.concat(fatias[:k]).reset_index(drop=True)
        teste = fatias[k].reset_index(drop=True)
        if len(treino) < 50 or len(teste) < 10:
            continue
        e_tr, cortes = discretizar(treino, q=q)
        stats = estatisticas_por_estado(treino, e_tr)
        if stats.empty:
            continue
        teto = treino["lote_max"].quantile(0.99)
        e_te, _ = discretizar(teste, cortes=cortes)
        if embaralhar_estados:                        # H0: estado nao informa o futuro
            e_te = pd.Series(rng.permutation(e_te.to_numpy()), index=e_te.index)
        sinais = gerar_sinais(teste, e_te, stats, limiar=limiar, teto_lote=teto)
        t = backtest(sinais, k_stop=k_stop, k_alvo=k_alvo, max_hold=max_hold, custo=custo)
        if not t.empty:
            t["bloco"] = k
            todos.append(t)
    return pd.concat(todos, ignore_index=True) if todos else pd.DataFrame()


def busca_parametros(barras_por_janela: dict[str, pd.DataFrame], grade: dict, n_blocos: int = 5,
                     custo: float = 0.0002, min_trades: int = 30, chave: str = "data") -> pd.DataFrame:
    """Grid search com avaliacao walk-forward. Retorna TODAS as combinacoes, nao so a melhor."""
    chaves = list(grade)
    linhas = []
    for combo in itertools.product(*(grade[k] for k in chaves)):
        p = dict(zip(chaves, combo))
        barras = barras_por_janela[p["janela"]]
        t = walk_forward(barras, n_blocos=n_blocos, custo=custo, chave=chave,
                         **{k: v for k, v in p.items() if k != "janela"})
        linhas.append({**p, **metricas(t)})
    r = pd.DataFrame(linhas)
    r["valido"] = r["trades"] >= min_trades
    return r.sort_values(["valido", "profit_factor"], ascending=False).reset_index(drop=True)


def teste_de_selecao(barras: pd.DataFrame, melhor: dict, n_perm: int = 100, n_blocos: int = 5,
                     custo: float = 0.0002, chave: str = "data") -> dict:
    """Quanto do resultado da melhor combinacao sobrevive quando o sinal e destruido.

    Roda a MESMA estrategia com os estados embaralhados. Se o PF real nao ficar
    acima do percentil 95 do nulo, o que voce otimizou foi ruido.
    """
    rng = np.random.default_rng(42)
    p = {k: v for k, v in melhor.items() if k in {"q", "limiar", "k_stop", "k_alvo", "max_hold"}}
    real = metricas(walk_forward(barras, n_blocos=n_blocos, custo=custo, chave=chave, **p))
    nulos = []
    for _ in range(n_perm):
        m = metricas(walk_forward(barras, n_blocos=n_blocos, custo=custo, chave=chave,
                                  embaralhar_estados=True, rng=rng, **p))
        if m["trades"] > 0 and np.isfinite(m["profit_factor"]):
            nulos.append(m["profit_factor"])
    nulos = np.array(nulos) if nulos else np.array([np.nan])
    pf = real["profit_factor"]
    return {"pf_real": pf, "pf_nulo_mediana": float(np.nanmedian(nulos)),
            "pf_nulo_p95": float(np.nanpercentile(nulos, 95)),
            "p_valor": float(np.nanmean(nulos >= pf)), "n_permutacoes": len(nulos)}


# ============================================================================
# Orquestracao
# ============================================================================
def executar(barras: pd.DataFrame, limiar: float = 0.55, k_stop: float = 1.5, k_alvo: float = 2.5,
             max_hold: int = 5, custo: float = 0.0002, n_blocos: int = 5, chave: str = "data",
             out: str | None = None) -> dict:
    e, cortes = discretizar(barras)
    P = matriz_transicao(e, barras["sessao"])
    trades = walk_forward(barras, n_blocos=n_blocos, limiar=limiar, k_stop=k_stop,
                          k_alvo=k_alvo, max_hold=max_hold, custo=custo, chave=chave)
    r = {
        "barras": barras, "matriz_transicao": P, "estacionaria": distribuicao_estacionaria(P),
        "markov": teste_markov(e, barras["sessao"]), "stats_estado": estatisticas_por_estado(barras, e),
        "cortes_ofi": cortes, "perfil": perfil_intradiario(barras),
        "trades": trades, "metricas": metricas(trades), "por_ativo": resumo_por_ativo(barras),
        "por_hora": trades.groupby("hora")["retorno"].agg(["size", "mean", "sum"]) if not trades.empty else pd.DataFrame(),
    }
    if out:
        salvar(r, out)
    return r


def salvar(r: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    r["matriz_transicao"].to_csv(f"{out}/matriz_transicao.csv")
    r["stats_estado"].to_csv(f"{out}/stats_estado.csv")
    r["perfil"].to_csv(f"{out}/perfil_intradiario.csv")
    r["por_ativo"].to_csv(f"{out}/por_ativo.csv")
    pd.DataFrame([r["metricas"]]).to_csv(f"{out}/metricas.csv", index=False)
    for nome, tab in (("barras", r["barras"]), ("trades", r["trades"])):
        if tab is None or tab.empty:
            continue
        try:
            tab.to_parquet(f"{out}/{nome}.parquet", index=False)
        except ImportError:
            tab.to_csv(f"{out}/{nome}.csv", index=False)


def dados_sinteticos(n_ativos: int = 6, n_sessoes: int = 5, ticks_por_sessao: int = 12_000,
                     seed: int = 7) -> pd.DataFrame:
    """Ticks simulados SEM sinal explotavel - testa o encanamento, nao a estrategia."""
    rng = np.random.default_rng(seed)
    partes = []
    for a in range(n_ativos):
        for k in range(n_sessoes):
            n = ticks_por_sessao
            s, sinal = 1, np.empty(n, dtype="int8")
            for i in range(n):
                if rng.random() < 0.15:
                    s = -s
                sinal[i] = s
            px = 20 + 5 * a + np.cumsum(sinal * rng.gamma(1.0, 0.004, n) - 0.0005 * rng.standard_normal(n))
            dia = pd.Timestamp("2026-03-02") + pd.Timedelta(days=k)
            ts = dia + pd.Timedelta(hours=13) + pd.to_timedelta(np.cumsum(rng.exponential(1.8, n)), unit="s")
            partes.append(pd.DataFrame({
                "timestamp": ts, "md_entry_px": px.round(2),
                "md_entry_size": rng.integers(1, 40, n) * np.where(rng.random(n) < 0.002, 50, 1),
                "aggressor": np.where(sinal > 0, 1, 2),
                "tick_direction": np.where(sinal > 0, 0, 2),
                "ativo": f"TEST{a}",
            }))
    return preparar(pd.concat(partes, ignore_index=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline BTG-TLD-A26 v2")
    p.add_argument("--path", help="arquivo ou diretorio parquet")
    p.add_argument("--freq", default="1min")
    p.add_argument("--ticks", type=int, help="barras de N ticks (ignora --freq)")
    p.add_argument("--pregao", help="faixa HH:MM-HH:MM, ex: 10:05-16:50")
    p.add_argument("--limiar", type=float, default=0.55)
    p.add_argument("--k-stop", type=float, default=1.5, help="stop em multiplos da sigma intradiaria")
    p.add_argument("--k-alvo", type=float, default=2.5, help="alvo em multiplos da sigma intradiaria")
    p.add_argument("--max-hold", type=int, default=5)
    p.add_argument("--custo", type=float, default=0.0002)
    p.add_argument("--blocos", type=int, default=5)
    p.add_argument("--tune", action="store_true", help="grid search walk-forward + teste de permutacao")
    p.add_argument("--perm", type=int, default=100)
    p.add_argument("--ticker", help="filtra por papel, ex: PETR4 ou PETR4,VALE3,ITUB4")
    p.add_argument("--datas", help="faixa AAAA-MM-DD:AAAA-MM-DD")
    p.add_argument("--chave", default="data", choices=["data", "ativo", "sessao"],
                   help="dimensao do walk-forward: data (painel) ou ativo (papeis novos)")
    p.add_argument("--schema", action="store_true", help="so inspeciona o dataset e sai")
    p.add_argument("--diag", action="store_true", help="infere a convencao do campo aggressor e sai")
    p.add_argument("--mapa", help='convencao do aggressor, ex: "3:1,4:-1"')
    p.add_argument("--memoria", action="store_true", help="carrega tudo na RAM em vez de por arquivo")
    p.add_argument("--out")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    if not a.demo and not a.path:
        p.error("informe --path <arquivo_ou_diretorio_parquet> ou use --demo")
    if a.path and not os.path.exists(a.path):
        p.error(f"caminho nao encontrado: {a.path}")
    pregao = tuple(a.pregao.split("-")) if a.pregao else None
    datas = tuple(a.datas.split(":")) if a.datas else None
    freq, ticks = (None, a.ticks) if a.ticks else (a.freq, None)

    if a.mapa:
        aplicar_mapa(a.mapa)
    if a.schema:
        inspecionar(a.path)
        return
    if a.diag:
        diagnosticar(a.path, ticker=a.ticker)
        return

    if a.demo:
        barras = construir_barras(dados_sinteticos(), freq=freq, ticks=ticks, derivar=True)
    elif a.memoria:
        df = carregar_ticks(a.path)
        if pregao:
            df = filtrar_pregao(df, *pregao)
        barras = construir_barras(df, freq=freq, ticks=ticks, derivar=True)
    else:
        barras = barras_multiativo(a.path, freq=freq, ticks=ticks, ticker=a.ticker,
                                   datas=datas, pregao=pregao)

    pd.set_option("display.width", 130)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    if a.tune:
        janelas = {"1min": barras} if not a.ticks else {f"{a.ticks}t": barras}
        grade = {"janela": list(janelas), "q": [(0.25, 0.75), (0.33, 0.67), (0.10, 0.90)],
                 "limiar": [0.52, 0.55, 0.58], "k_stop": [1.0, 1.5, 2.0],
                 "k_alvo": [1.5, 2.5], "max_hold": [3, 5, 10]}
        r = busca_parametros(janelas, grade, n_blocos=a.blocos, custo=a.custo, chave=a.chave)
        print("\n--- grid walk-forward (top 10 de %d combinacoes) ---" % len(r))
        print(r.head(10).to_string(index=False))
        melhor = r[r["valido"]].iloc[0].to_dict() if r["valido"].any() else r.iloc[0].to_dict()
        print("\n--- teste de permutacao da melhor combinacao ---")
        for k, v in teste_de_selecao(barras, melhor, n_perm=a.perm, n_blocos=a.blocos,
                                     custo=a.custo, chave=a.chave).items():
            print(f"  {k}: {v}")
        print(f"\n  combinacoes testadas: {len(r)} -> exija p_valor bem abaixo de {0.05/max(len(r),1):.4f} (Bonferroni)")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            r.to_csv(f"{a.out}/grid.csv", index=False)
        return

    r = executar(barras, limiar=a.limiar, k_stop=a.k_stop, k_alvo=a.k_alvo, max_hold=a.max_hold,
                 custo=a.custo, n_blocos=a.blocos, chave=a.chave, out=a.out)
    print(f"\nbarras: {len(barras):,} | ativos: {barras['ativo'].nunique()} | "
          f"pregoes: {barras['data'].nunique()} | cortes OFI: {r['cortes_ofi']}")
    print("\n--- matriz de transicao ---\n", r["matriz_transicao"])
    print("\n--- teste de Markov (independencia) ---\n ", r["markov"])
    print("\n--- estatisticas por estado ---\n", r["stats_estado"])
    print("\n--- perfil intradiario ---\n", r["perfil"])
    print("\n--- top/bottom ativos por poder preditivo do OFI ---")
    pa = r["por_ativo"]
    print(pd.concat([pa.head(10), pa.tail(5)]).to_string())
    print("\n--- walk-forward (fora da amostra) ---")
    for k, v in r["metricas"].items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if not r["por_hora"].empty:
        print("\n--- P&L por hora ---\n", r["por_hora"])


if __name__ == "__main__":
    main()
