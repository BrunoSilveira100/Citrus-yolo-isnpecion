"""Gera frontend/docs/imagens/fluxo_simulado.jpg: simulação da inspeção na linha de beneficiamento.

As frutas são fotos reais do dataset e a classe/confiança de cada uma vem do modelo treinado neste
projeto (backend/modelo/best.pt). A cena (câmera, esteira, destinos e painel) é ilustrativa: a
integração com a linha é a fase 3 do roadmap.

Uso (da raiz do repositório, com a .venv do backend):
    backend/.venv/Scripts/python.exe frontend/docs/gerar_fluxo_simulado.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAIZ = Path(__file__).resolve().parents[2]
BACK = RAIZ / "backend"
SAIDA = RAIZ / "frontend" / "docs" / "imagens" / "fluxo_simulado.jpg"

W, H = 1600, 900
FUNDO = (17, 21, 27)
CARTAO = (26, 31, 39)
LINHA = (52, 59, 70)
TEXTO = (232, 234, 237)
APAGADO = (140, 148, 160)
VERDE = (64, 192, 110)
VERMELHO = (232, 76, 76)
AMBAR = (240, 170, 50)
CINZA = (120, 126, 136)

# (arquivo, posição x na esteira). As 2 primeiras ainda não passaram pela câmera.
FRUTAS = [
    ("dataset/images/val/healthy_Healthy 157.jpg", 105, False),
    ("dataset/images/val/black_spot_Black spot 150.jpg", 230, False),
    ("dataset/ineditas/greening_Greening 59.jpg", 400, True),  # sob a câmera
    ("dataset/images/val/healthy_Healthy 111.jpg", 545, True),
    ("dataset/images/val/black_spot_Black spot 142.jpg", 665, True),
    ("dataset/images/val/healthy_Healthy 151.jpg", 785, True),
    ("dataset/ineditas/healthy_Healthy 61.jpg", 905, True),
]
CENTRO_ESTEIRA = 585
DIAM = 104
CAMERA_X = 400


def fonte(tamanho: int, negrito: bool = False) -> ImageFont.FreeTypeFont:
    nomes = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if negrito else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for nome in nomes:
        for pasta in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            if (pasta / nome).is_file():
                return ImageFont.truetype(str(pasta / nome), tamanho)
    return ImageFont.load_default()


def classificar(caminhos: list[Path]) -> list[dict]:
    from ultralytics import YOLO

    modelo = YOLO(str(BACK / "modelo" / "best.pt"))
    saida = []
    for r in modelo.predict([str(c) for c in caminhos], conf=0.1, imgsz=640, verbose=False):
        i = int(r.boxes.conf.argmax())
        conf = float(r.boxes.conf[i])
        classe = modelo.names[int(r.boxes.cls[i])]
        destino = "revisao" if conf < 0.5 else ("descarte" if classe == "laranja_defeito" else "embalagem")
        saida.append({"img": Image.open(r.path).convert("RGB"), "xyxy": r.boxes.xyxy[i].tolist(),
                      "classe": classe, "conf": conf, "destino": destino})
    return saida


def fruta_redonda(info: dict, diam: int) -> Image.Image:
    """Recorte quadrado em volta da fruta detectada, com máscara circular (parece fruta na esteira)."""
    x1, y1, x2, y2 = info["xyxy"]
    cx, cy, lado = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.98
    recorte = info["img"].crop((cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2)).resize((diam, diam))
    mascara = Image.new("L", (diam, diam), 0)
    ImageDraw.Draw(mascara).ellipse((1, 1, diam - 2, diam - 2), fill=255)
    recorte.putalpha(mascara.filter(ImageFilter.GaussianBlur(1)))
    return recorte


def cor_destino(destino: str) -> tuple[int, int, int]:
    return {"embalagem": VERDE, "descarte": VERMELHO, "revisao": AMBAR}[destino]


def rotulo(info: dict) -> str:
    nome = "boa" if info["classe"] == "laranja_boa" else "defeito"
    return f"{nome} {info['conf']:.2f}".replace(".", ",")


def etiqueta(d: ImageDraw.ImageDraw, x: int, y: int, texto: str, cor, f) -> None:
    caixa = d.textbbox((0, 0), texto, font=f)
    w, h = caixa[2] - caixa[0], caixa[3] - caixa[1]
    d.rounded_rectangle((x, y, x + w + 14, y + h + 10), radius=5, fill=cor)
    d.text((x + 7, y + 3 - caixa[1]), texto, fill=(15, 15, 15), font=f)


def seta(d: ImageDraw.ImageDraw, a, b, cor, largura=3) -> None:
    d.line([a, b], fill=cor, width=largura)
    v = np.array(b, float) - np.array(a, float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    ponta = np.array(b, float)
    d.polygon([tuple(ponta), tuple(ponta - 14 * v + 7 * n), tuple(ponta - 14 * v - 7 * n)], fill=cor)


def main() -> None:
    infos = classificar([BACK / f for f, _, _ in FRUTAS])
    tela = Image.new("RGB", (W, H), FUNDO)
    d = ImageDraw.Draw(tela, "RGBA")
    f_tit, f_sub, f_txt, f_peq = fonte(34, True), fonte(19), fonte(18, True), fonte(15)
    f_kpi, f_cab = fonte(40, True), fonte(16, True)

    d.text((40, 26), "Simulação: inspeção de laranjas na linha de beneficiamento", fill=TEXTO, font=f_tit)
    d.text((40, 74), "Fotos reais do dataset · classe e confiança dadas pelo modelo treinado neste projeto",
           fill=APAGADO, font=f_sub)

    # ---- esteira
    topo, base = CENTRO_ESTEIRA - 72, CENTRO_ESTEIRA + 72
    d.rectangle((30, topo, 1000, base), fill=(48, 53, 61))
    for x in range(30, 1000, 26):
        d.line([(x, topo + 6), (x, base - 6)], fill=(40, 44, 51), width=8)
    d.rectangle((30, topo - 10, 1000, topo), fill=(88, 95, 106))
    d.rectangle((30, base, 1000, base + 10), fill=(88, 95, 106))
    seta(d, (560, base + 34), (720, base + 34), APAGADO, 2)
    d.text((732, base + 22), "sentido da esteira", fill=APAGADO, font=f_peq)

    # ---- câmera, suporte e cone de luz
    d.rectangle((CAMERA_X - 4, 112, CAMERA_X + 4, 250), fill=(90, 96, 106))
    d.polygon([(CAMERA_X - 22, 300), (CAMERA_X + 22, 300), (CAMERA_X + 78, topo - 12), (CAMERA_X - 78, topo - 12)],
              fill=(255, 220, 120, 38))
    d.rounded_rectangle((CAMERA_X - 46, 240, CAMERA_X + 46, 292), radius=8, fill=(38, 44, 54), outline=(110, 118, 130), width=2)
    d.ellipse((CAMERA_X - 18, 270, CAMERA_X + 18, 306), fill=(20, 24, 30), outline=(150, 160, 175), width=3)
    d.ellipse((CAMERA_X - 7, 281, CAMERA_X + 7, 295), fill=(70, 130, 200))
    d.text((CAMERA_X - 170, 150), "câmera +", fill=APAGADO, font=f_peq)
    d.text((CAMERA_X - 170, 170), "iluminação", fill=APAGADO, font=f_peq)

    # ---- frutas na esteira
    for (_, x, classificada), info in zip(FRUTAS, infos, strict=True):
        d.ellipse((x - DIAM / 2 + 6, CENTRO_ESTEIRA - DIAM / 2 + 12, x + DIAM / 2 + 6, CENTRO_ESTEIRA + DIAM / 2 + 12),
                  fill=(0, 0, 0, 90))
        tela.paste(fruta_redonda(info, DIAM), (x - DIAM // 2, CENTRO_ESTEIRA - DIAM // 2), fruta_redonda(info, DIAM))
        if classificada:
            cor = cor_destino(info["destino"])
            m = DIAM // 2 + 6
            d.rectangle((x - m, CENTRO_ESTEIRA - m, x + m, CENTRO_ESTEIRA + m), outline=cor, width=3)
            etiqueta(d, x - m, CENTRO_ESTEIRA - m - 30, rotulo(info), cor, f_peq)
        else:
            d.text((x - 38, base + 16), "aguardando", fill=APAGADO, font=f_peq)

    # ---- frame capturado (fruta sob a câmera)
    alvo = infos[2]
    fx1, fy1, fx2, fy2 = 560, 128, 1000, 452
    d.rounded_rectangle((fx1, fy1, fx2, fy2), radius=10, fill=CARTAO, outline=LINHA, width=2)
    d.text((fx1 + 16, fy1 + 12), "FRAME CAPTURADO  →  YOLOv8n", fill=APAGADO, font=f_cab)
    foto = alvo["img"].copy()
    escala = min((fx2 - fx1 - 32) / foto.width, (fy2 - fy1 - 56) / foto.height)
    foto = foto.resize((int(foto.width * escala), int(foto.height * escala)))
    px, py = fx1 + (fx2 - fx1 - foto.width) // 2, fy1 + 44
    tela.paste(foto, (px, py))
    bx1, by1, bx2, by2 = (v * escala for v in alvo["xyxy"])
    cor = cor_destino(alvo["destino"])
    d.rectangle((px + bx1, py + by1, px + bx2, py + by2), outline=cor, width=3)
    etiqueta(d, int(px + bx1), int(py + by1) + 4, rotulo(alvo), cor, f_peq)
    seta(d, (CAMERA_X + 50, 266), (fx1 - 6, 266), APAGADO, 2)

    # ---- destinos
    rotas = [("embalagem", "EMBALAGEM", "boa e confiança ≥ 0,50"), ("descarte", "DESCARTE", "defeito"),
             ("revisao", "REVISÃO HUMANA", "confiança < 0,50")]
    classificadas = [i for (_, _, c), i in zip(FRUTAS, infos, strict=True) if c]
    for k, (chave, titulo, regra) in enumerate(rotas):
        y1 = 128 + k * 200
        cor = cor_destino(chave)
        d.rounded_rectangle((1080, y1, 1560, y1 + 176), radius=10, fill=CARTAO, outline=cor, width=2)
        d.rectangle((1080, y1 + 10, 1086, y1 + 166), fill=cor)
        d.text((1104, y1 + 14), titulo, fill=cor, font=f_txt)
        d.text((1104, y1 + 40), regra, fill=APAGADO, font=f_peq)
        itens = [i for i in classificadas if i["destino"] == chave]
        d.text((1500, y1 + 12), str(len(itens)), fill=TEXTO, font=f_kpi)
        for j, info in enumerate(itens):
            mini = fruta_redonda(info, 76)
            tela.paste(mini, (1104 + j * 96, y1 + 76), mini)
            d.text((1104 + j * 96, y1 + 154), rotulo(info), fill=APAGADO, font=fonte(13))
        seta(d, (1006, CENTRO_ESTEIRA), (1074, y1 + 88), cor, 3)

    # ---- painel do lote
    total = len(classificadas)
    defeitos = sum(i["destino"] == "descarte" for i in classificadas)
    revisao = sum(i["destino"] == "revisao" for i in classificadas)
    d.rounded_rectangle((30, 712, 1000, 868), radius=10, fill=CARTAO, outline=LINHA, width=2)
    d.text((52, 726), "PAINEL DO LOTE", fill=APAGADO, font=f_cab)
    kpis = [("INSPECIONADAS", str(total), TEXTO), ("DEFEITOS", str(defeitos), VERMELHO),
            ("TAXA DE DEFEITO", f"{100 * defeitos / total:.0f}%", TEXTO), ("EM REVISÃO", str(revisao), AMBAR)]
    for k, (nome, valor, cor) in enumerate(kpis):
        x = 52 + k * 238
        d.text((x, 762), nome, fill=APAGADO, font=f_peq)
        d.text((x, 786), valor, fill=cor, font=f_kpi)

    d.text((1080, 740), "Simulação ilustrativa da fase 3 do roadmap.", fill=APAGADO, font=f_peq)
    d.text((1080, 762), "Câmera e esteira são desenho; as frutas e as", fill=APAGADO, font=f_peq)
    d.text((1080, 784), "classificações são reais (modelo deste projeto).", fill=APAGADO, font=f_peq)
    d.text((1080, 818), "Fotos: Orange Fruit Diseases Dataset (Doh, 2025).", fill=CINZA, font=f_peq)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tela.save(SAIDA, quality=88)
    for i in classificadas:
        print(i["classe"], round(i["conf"], 3), i["destino"])
    print("salvo em", SAIDA)


if __name__ == "__main__":
    main()
