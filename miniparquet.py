"""Leitor minimo de Parquet em Python puro (sem pyarrow/fastparquet).

Cobre o necessario para o BTG-TLD-A26: thrift compact protocol para o footer,
descompressao SNAPPY, e decodificacao PLAIN / RLE_DICTIONARY / RLE-bitpacked.
"""
from __future__ import annotations

import struct

# ---------------------------------------------------------------- thrift compact
STOP, TRUE, FALSE, BYTE, I16, I32, I64, DOUBLE, BINARY, LIST, SET, MAP, STRUCT = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


class Thrift:
    def __init__(self, buf: bytes, pos: int = 0):
        self.b, self.p = buf, pos

    def byte(self) -> int:
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self) -> int:
        r = s = 0
        while True:
            c = self.b[self.p]
            self.p += 1
            r |= (c & 0x7F) << s
            if not c & 0x80:
                return r
            s += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self) -> bytes:
        n = self.varint()
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def struct(self) -> dict:
        """Retorna {field_id: valor}. Structs aninhados viram dicts, listas viram list."""
        out, fid = {}, 0
        while True:
            h = self.byte()
            if h == STOP:
                return out
            delta, ttype = h >> 4, h & 0x0F
            fid = fid + delta if delta else self.zigzag()
            out[fid] = self.value(ttype)

    def value(self, t: int):
        if t == TRUE:
            return True
        if t == FALSE:
            return False
        if t == BYTE:
            return self.byte()
        if t in (I16, I32, I64):
            return self.zigzag()
        if t == DOUBLE:
            v = struct.unpack("<d", self.b[self.p:self.p + 8])[0]
            self.p += 8
            return v
        if t == BINARY:
            return self.binary()
        if t == STRUCT:
            return self.struct()
        if t in (LIST, SET):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            return [self.value(et) for _ in range(n)]
        if t == MAP:
            n = self.varint()
            if n == 0:
                return {}
            kt = self.byte()
            return {self.value(kt >> 4): self.value(kt & 0x0F) for _ in range(n)}
        raise ValueError(f"tipo thrift desconhecido: {t}")


# ---------------------------------------------------------------- snappy (raw)
def snappy_decompress(data: bytes) -> bytes:
    p, n = 0, len(data)
    # tamanho descomprimido (varint)
    ln = shift = 0
    while True:
        c = data[p]
        p += 1
        ln |= (c & 0x7F) << shift
        if not c & 0x80:
            break
        shift += 7
    out = bytearray()
    while p < n:
        tag = data[p]
        p += 1
        t = tag & 0x03
        if t == 0:                                  # literal
            ln_lit = tag >> 2
            if ln_lit < 60:
                ln_lit += 1
            else:
                nb = ln_lit - 59
                ln_lit = int.from_bytes(data[p:p + nb], "little") + 1
                p += nb
            out += data[p:p + ln_lit]
            p += ln_lit
        else:
            if t == 1:                              # copy, offset 11 bits
                length = 4 + ((tag >> 2) & 0x07)
                off = ((tag >> 5) << 8) | data[p]
                p += 1
            elif t == 2:                            # copy, offset 16 bits
                length = (tag >> 2) + 1
                off = int.from_bytes(data[p:p + 2], "little")
                p += 2
            else:                                   # copy, offset 32 bits
                length = (tag >> 2) + 1
                off = int.from_bytes(data[p:p + 4], "little")
                p += 4
            start = len(out) - off
            for i in range(length):                 # sobreposicao e permitida
                out.append(out[start + i])
    if len(out) != ln:
        raise ValueError("snappy: tamanho inesperado")
    return bytes(out)


def descomprimir(codec: int, data: bytes, tam: int) -> bytes:
    # CompressionCodec: 0 NONE, 1 SNAPPY, 2 GZIP, 6 ZSTD, 7 LZ4_RAW
    if codec == 0:
        return data
    if codec == 1:
        return snappy_decompress(data)
    if codec == 2:
        import gzip
        return gzip.decompress(data)
    raise ValueError(f"codec {codec} nao suportado neste leitor")


# ---------------------------------------------------------------- bit unpack / RLE
def bit_unpack(buf: bytes, pos: int, width: int, count: int) -> tuple[list[int], int]:
    """Bit-packing little-endian do Parquet (grupos de 8 valores)."""
    vals = []
    if width == 0:
        return [0] * count, pos
    nbytes = (count * width + 7) // 8
    acc = int.from_bytes(buf[pos:pos + nbytes], "little")
    mask = (1 << width) - 1
    for i in range(count):
        vals.append((acc >> (i * width)) & mask)
    return vals, pos + nbytes


def rle_decode(buf: bytes, pos: int, width: int, n_values: int, limite: int | None = None) -> tuple[list[int], int]:
    """Hibrido RLE/bit-packed do Parquet."""
    out = []
    fim = limite if limite is not None else len(buf)
    while len(out) < n_values and pos < fim:
        t = Thrift(buf, pos)
        header = t.varint()
        pos = t.p
        if header & 1:                              # bit-packed
            grupos = header >> 1
            count = grupos * 8
            vals, pos = bit_unpack(buf, pos, width, count)
            out.extend(vals)
        else:                                       # RLE
            rep = header >> 1
            nb = (width + 7) // 8
            v = int.from_bytes(buf[pos:pos + nb], "little")
            pos += nb
            out.extend([v] * rep)
    return out[:n_values], pos


def bits_needed(n: int) -> int:
    return max(n - 1, 0).bit_length()


# ---------------------------------------------------------------- PLAIN
TIPOS = {0: "BOOLEAN", 1: "INT32", 2: "INT64", 3: "INT96", 4: "FLOAT", 5: "DOUBLE", 6: "BYTE_ARRAY",
         7: "FIXED_LEN_BYTE_ARRAY"}


def plain_decode(buf: bytes, tipo: int, count: int, tam_fixo: int = 0) -> list:
    if tipo == 1:
        return list(struct.unpack(f"<{count}i", buf[:4 * count]))
    if tipo == 2:
        return list(struct.unpack(f"<{count}q", buf[:8 * count]))
    if tipo == 4:
        return list(struct.unpack(f"<{count}f", buf[:4 * count]))
    if tipo == 5:
        return list(struct.unpack(f"<{count}d", buf[:8 * count]))
    if tipo == 6:
        out, p = [], 0
        for _ in range(count):
            n = struct.unpack("<I", buf[p:p + 4])[0]
            p += 4
            out.append(buf[p:p + n])
            p += n
        return out
    if tipo == 7:
        return [buf[i * tam_fixo:(i + 1) * tam_fixo] for i in range(count)]
    if tipo == 0:
        out = []
        for i in range(count):
            out.append(bool((buf[i // 8] >> (i % 8)) & 1))
        return out
    raise ValueError(f"tipo {tipo} nao suportado")


# ---------------------------------------------------------------- arquivo
class ParquetFile:
    def __init__(self, caminho: str):
        self.caminho = caminho
        with open(caminho, "rb") as f:
            f.seek(-8, 2)
            n = struct.unpack("<I", f.read(4))[0]
            f.seek(-8 - n, 2)
            footer = f.read(n)
        self.meta = Thrift(footer).struct()
        self.n_rows = self.meta[3]
        self.schema = self.meta[2]
        self.row_groups = self.meta[4]
        self.colunas = {}
        for el in self.schema[1:]:
            nome = el[4].decode() if isinstance(el[4], bytes) else el[4]
            self.colunas[nome] = {"tipo": el.get(1), "repetition": el.get(3),
                                  "logical": el.get(10), "converted": el.get(6),
                                  "tam_fixo": el.get(7, 0)}

    @property
    def nomes(self) -> list[str]:
        return list(self.colunas)

    def ler_coluna(self, nome: str) -> list:
        if nome not in self.colunas:
            raise KeyError(f"coluna {nome} ausente; disponiveis: {self.nomes}")
        valores = []
        with open(self.caminho, "rb") as f:
            for rg in self.row_groups:
                for cc in rg[1]:
                    meta = cc[3]
                    caminho_col = [c.decode() if isinstance(c, bytes) else c for c in meta[3]]
                    if caminho_col[-1] != nome:
                        continue
                    valores.extend(self._ler_chunk(f, meta))
        return valores

    def _ler_chunk(self, f, meta) -> list:
        tipo, codec, n_vals = meta[1], meta[4], meta[5]
        # ColumnMetaData: 9 data_page_offset, 11 dictionary_page_offset, 7 total_compressed_size
        inicio = meta[9]
        if meta.get(11):
            inicio = min(inicio, meta[11])
        total_comp = meta[7]
        f.seek(inicio)
        bloco = f.read(total_comp)

        p, saida, dicionario = 0, [], None
        opcional = self.colunas[[k for k in self.colunas][0]] is not None  # placeholder
        rep = self.colunas.get(meta[3][-1].decode() if isinstance(meta[3][-1], bytes) else meta[3][-1], {})
        opcional = rep.get("repetition", 1) == 1

        while p < len(bloco) and len(saida) < n_vals:
            t = Thrift(bloco, p)
            ph = t.struct()
            p = t.p
            tipo_pg, tam_desc, tam_comp = ph[1], ph[2], ph[3]
            dados = descomprimir(codec, bloco[p:p + tam_comp], tam_desc)
            p += tam_comp

            if tipo_pg == 2:                         # dictionary page
                n = ph[7][1]
                dicionario = plain_decode(dados, tipo, n, self.colunas[
                    meta[3][-1].decode() if isinstance(meta[3][-1], bytes) else meta[3][-1]].get("tam_fixo", 0))
                continue

            if tipo_pg == 0:                         # data page v1
                hdr = ph[5]
                n = hdr[1]
                enc, enc_def = hdr[2], hdr[4]
                q = 0
                if opcional:
                    tam_def = struct.unpack("<I", dados[:4])[0]
                    defs, _ = rle_decode(dados[4:4 + tam_def], 0, 1, n)
                    q = 4 + tam_def
                else:
                    defs = [1] * n
                presentes = sum(defs)
                corpo = dados[q:]
                if enc in (2, 8):                    # RLE_DICTIONARY / PLAIN_DICTIONARY
                    width = corpo[0]
                    idx, _ = rle_decode(corpo, 1, width, presentes)
                    vals = [dicionario[i] for i in idx]
                else:
                    vals = plain_decode(corpo, tipo, presentes)
                it = iter(vals)
                saida.extend([next(it) if d else None for d in defs])
            elif tipo_pg == 3:                       # data page v2
                hdr = ph[8]
                n, n_nulos = hdr[1], hdr[3]
                tam_def = hdr[6]
                defs, _ = rle_decode(dados[:tam_def], 0, 1, n) if tam_def else ([1] * n, 0)
                corpo = dados[tam_def:]
                presentes = n - n_nulos
                if hdr[4] in (2, 8):
                    width = corpo[0]
                    idx, _ = rle_decode(corpo, 1, width, presentes)
                    vals = [dicionario[i] for i in idx]
                else:
                    vals = plain_decode(corpo, tipo, presentes)
                it = iter(vals)
                saida.extend([next(it) if d else None for d in defs])
            else:
                raise ValueError(f"pagina tipo {tipo_pg} nao suportada")
        return saida[:n_vals]

    def to_dict(self, colunas: list[str] | None = None) -> dict:
        return {c: self.ler_coluna(c) for c in (colunas or self.nomes)}
