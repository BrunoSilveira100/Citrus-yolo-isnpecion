// Gera frontend/apresentacao/apresentacao_projeto.docx no padrão ABNT
// (NBR 14724 estrutura, NBR 6024 numeração, NBR 6023 referências, NBR 10520 citações, IBGE tabelas).
// Uso (da raiz do repositório):
//   node frontend/apresentacao/gerar_apresentacao.js frontend/apresentacao/apresentacao_projeto.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  AlignmentType, HeadingLevel, PageBreak, LevelFormat, BorderStyle, Header, PageNumber, TableOfContents,
  TabStopType,
} = require("docx");

// ---------------------------------------------------------------- formato ABNT
const FONTE = "Arial";
const T12 = 24; // 12 pt (meio-pontos)
const T10 = 20; // 10 pt: legendas, fontes, tabelas, natureza do trabalho
const CM = 567; // 1 cm em DXA
const MARGENS = { top: 3 * CM, left: 3 * CM, bottom: 2 * CM, right: 2 * CM };
const LARGURA = 11906 - MARGENS.left - MARGENS.right; // A4 menos margens
const RECUO = Math.round(1.25 * CM);
const LINHA_15 = 360;
const RES = (f) => path.join(__dirname, "../../backend/resultados", f);
const PENDENTE = "yellow"; // destaque para o que o aluno ainda precisa preencher

const run = (t, o = {}) => new TextRun({ text: t, font: FONTE, size: T12, ...o });
const partes = (conteudo, o = {}) =>
  (Array.isArray(conteudo) ? conteudo : [conteudo]).map((c) => (typeof c === "string" ? run(c, o) : run(c.t, { ...o, ...c })));

// parágrafo de texto: justificado, 1,5, recuo de 1,25 cm na primeira linha
const p = (conteudo, o = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { line: LINHA_15, after: 0 },
  indent: { firstLine: RECUO }, ...o, children: partes(conteudo),
});
const centro = (conteudo, o = {}, r = {}) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { line: LINHA_15, after: 0 }, ...o, children: partes(conteudo, r),
});
const vazio = (n = 1) => Array.from({ length: n }, () => new Paragraph({ spacing: { line: LINHA_15 }, children: [] }));
const quebra = () => new Paragraph({ children: [new PageBreak()] });

// títulos: seção primária em MAIÚSCULAS e negrito, sempre em página nova (NBR 14724)
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [run(t, { bold: true })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(t, { bold: true })] });
// títulos sem indicativo numérico (centralizados)
const h1SemNumero = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER, children: [run(t, { bold: true })],
});
const tituloCentral = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { line: LINHA_15, after: LINHA_15 }, children: [run(t, { bold: true })],
});

// alíneas: a) b) c) ... (NBR 6024)
let instanciaAlinea = 0;
const alineas = (itens) => {
  instanciaAlinea += 1;
  return itens.map((t) => new Paragraph({
    numbering: { reference: "alineas", level: 0, instance: instanciaAlinea },
    alignment: AlignmentType.JUSTIFIED, spacing: { line: LINHA_15 }, children: partes(t),
  }));
};

// bloco de código: as linhas ficam juntas na mesma página
const codigo = (linhas) => linhas.map((l, i) => new Paragraph({
  spacing: { line: 240, after: 0 }, indent: { left: RECUO }, keepNext: i < linhas.length - 1, keepLines: true,
  children: [new TextRun({ text: l, font: "Consolas", size: T10 })],
}));

// ilustrações: identificação ACIMA, fonte ABAIXO (NBR 14724)
let nFigura = 0;
let nTabela = 0;
let nQuadro = 0;
const legenda = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 60, line: 240 }, keepNext: true,
  children: [run(t, { size: T10 })],
});
const fonte = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 240 }, children: [run(`Fonte: ${t}`, { size: T10 })],
});

function figura(titulo, arquivo, larguraCm, larguraPx, alturaPx, origem) {
  nFigura += 1;
  const w = Math.round(larguraCm * 37.8); // cm -> px a 96 dpi
  return [
    legenda(`Figura ${nFigura} – ${titulo}`),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [
      new ImageRun({ type: arquivo.endsWith(".png") ? "png" : "jpg", data: fs.readFileSync(arquivo),
        transformation: { width: w, height: Math.round((w * alturaPx) / larguraPx) } }),
    ] }),
    fonte(origem),
  ];
}

const SEM = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const LINHA = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
function celula(texto, largura, { negrito = false, bordas, alinhar = AlignmentType.CENTER } = {}) {
  return new TableCell({
    width: { size: largura, type: WidthType.DXA }, borders: bordas,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: alinhar, spacing: { line: 240 }, children: [run(texto, { size: T10, bold: negrito })] })],
  });
}

// tabela no padrão IBGE: só traços horizontais (topo, sob o cabeçalho e base), laterais abertas
function tabela(titulo, cabecalho, linhas, larguras, origem, alinharPrimeira = AlignmentType.LEFT) {
  nTabela += 1;
  const ultima = linhas.length - 1;
  const t = new Table({
    width: { size: larguras.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: larguras,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: cabecalho.map((c, i) => celula(c, larguras[i], {
        negrito: true, bordas: { top: LINHA, bottom: LINHA, left: SEM, right: SEM } })) }),
      ...linhas.map((l, r) => new TableRow({ children: l.map((c, i) => celula(c, larguras[i], {
        alinhar: i === 0 ? alinharPrimeira : AlignmentType.CENTER,
        bordas: { top: SEM, bottom: r === ultima ? LINHA : SEM, left: SEM, right: SEM } })) })),
    ],
  });
  return [legenda(`Tabela ${nTabela} – ${titulo}`), t, fonte(origem)];
}

// quadro: conteúdo textual, todas as bordas fechadas
function quadro(titulo, cabecalho, linhas, larguras, origem) {
  nQuadro += 1;
  const b = { top: LINHA, bottom: LINHA, left: LINHA, right: LINHA };
  const t = new Table({
    width: { size: larguras.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: larguras,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: cabecalho.map((c, i) => new TableCell({
        width: { size: larguras[i], type: WidthType.DXA }, borders: b, margins: { top: 40, bottom: 40, left: 80, right: 80 },
        shading: { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run(c, { size: T10, bold: true })] })] })) }),
      ...linhas.map((l) => new TableRow({ children: l.map((c, i) => celula(c, larguras[i], {
        bordas: b, alinhar: AlignmentType.LEFT })) })),
    ],
  });
  return [legenda(`Quadro ${nQuadro} – ${titulo}`), t, fonte(origem)];
}

const AUTOR = "elaborado pelo autor (2026)";

// ---------------------------------------------------------------- elementos pré-textuais
const capa = [
  centro("[NOME DA INSTITUIÇÃO DE ENSINO]", {}, { bold: true, highlight: PENDENTE }),
  centro("[NOME DO CURSO]", {}, { bold: true, highlight: PENDENTE }),
  ...vazio(3),
  centro("BRUNO HENRIQUE DA SILVEIRA", {}, { bold: true }),
  ...vazio(4),
  centro("ATIVIDADE PRÁTICA FINAL", {}, { bold: true }),
  centro("INTELIGÊNCIA ARTIFICIAL", {}, { bold: true }),
  centro("Desenvolvimento e Treinamento de um Modelo de Reconhecimento de Objetos com YOLO"),
  ...vazio(2),
  ...[
    ["Disciplina", "Inteligência Artificial"],
    ["Modalidade", "Atividade prática avaliativa"],
    ["Valor", "10,0 pontos"],
    ["Realização", "Individual"],
    ["Entrega", "Projeto funcional, resultados e relatório técnico simplificado"],
    ["Prazo", "Conforme orientação do professor"],
  ].map(([k, v]) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 276 },
    children: [run(`${k}: `, { bold: true, size: 22 }), run(v, { size: 22 })] })),
  ...vazio(4),
  centro("[CIDADE]", {}, { highlight: PENDENTE }),
  centro("2026"),
];

const naturezaRecuo = { left: Math.round(8 * CM) };
const folhaDeRosto = [
  centro("BRUNO HENRIQUE DA SILVEIRA", {}, { bold: true }),
  ...vazio(5),
  centro("ATIVIDADE PRÁTICA FINAL: INTELIGÊNCIA ARTIFICIAL", {}, { bold: true }),
  centro("Desenvolvimento e Treinamento de um Modelo de Reconhecimento de Objetos com YOLO"),
  ...vazio(3),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: naturezaRecuo, spacing: { line: 240 }, children: [run(
    "Relatório técnico apresentado à disciplina de Inteligência Artificial como Atividade Prática Final " +
    "avaliativa, com o projeto funcional, os resultados e a análise do desenvolvimento e treinamento de um " +
    "modelo de detecção de objetos com a arquitetura YOLO.", { size: T10 })] }),
  new Paragraph({ indent: naturezaRecuo, spacing: { before: 240, line: 240 }, children: [
    run("Professor(a): ", { size: T10 }), run("[Nome do professor]", { size: T10, highlight: PENDENTE })] }),
  ...vazio(6),
  centro("[CIDADE]", {}, { highlight: PENDENTE }),
  centro("2026"),
];

const resumo = [
  quebra(),
  tituloCentral("RESUMO"),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 240 }, children: [run(
    "Este trabalho apresenta o desenvolvimento e o treinamento de um modelo de detecção de objetos com a " +
    "arquitetura YOLO para inspecionar laranjas e separá-las em duas classes: laranja boa e laranja com " +
    "defeito. A motivação vem do beneficiamento de citros, em que a seleção visual ainda depende muito do " +
    "olhar humano. Usei o Orange Fruit Diseases Dataset, um conjunto público com 700 fotos organizadas por " +
    "diagnóstico. Como o arquivo não trazia as caixas delimitadoras, criei uma anotação semi-automática, " +
    "removi fotos com várias frutas e quase-duplicatas e cheguei a 240 imagens equilibradas, divididas em " +
    "192 para treino e 48 para validação, além de 8 imagens inéditas reservadas para o teste final. O " +
    "modelo YOLOv8n, pré-treinado, foi ajustado por 30 épocas em CPU. Na validação, obteve precision de " +
    "0,899, recall de 0,978 e mAP50 de 0,986. Nas imagens inéditas, acertou 5 de 8 sem nenhum erro. A " +
    "diferença entre esses dois resultados foi a principal lição do trabalho: cópias giradas da mesma foto " +
    "deixaram a validação otimista, e as imagens inéditas mostraram o desempenho mais próximo do real. O " +
    "projeto foi organizado como um sistema com regras de qualidade de dados, testes automatizados e " +
    "geração de relatório sem números digitados à mão.")] }),
  new Paragraph({ spacing: { before: 360, line: 240 }, alignment: AlignmentType.JUSTIFIED, children: [
    run("Palavras-chave: ", { bold: true }),
    run("visão computacional; detecção de objetos; YOLO; inspeção de frutas; aprendizado de máquina.")] }),
];

const sumario = [
  quebra(),
  tituloCentral("SUMÁRIO"),
  new TableOfContents("Sumário", { hyperlink: true, headingStyleRange: "1-2" }),
];

// ---------------------------------------------------------------- elementos textuais
const textual = [
  // ------------------------------------------------------------ 1
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run("1 INTRODUÇÃO", { bold: true })] }),
  p("Até esta atividade, a disciplina trabalhou com modelos que já vinham treinados. Aqui o desafio era outro: ensinar um modelo a reconhecer algo novo a partir de exemplos. Escolhi a inspeção de laranjas porque é um problema concreto. No beneficiamento de citros, a fruta é lavada, selecionada e embalada, e a seleção ainda depende muito de pessoas olhando fruta por fruta na esteira. É um trabalho repetitivo, sujeito a cansaço, e uma laranja doente que passa como boa compromete o lote."),
  p("O resultado é um sistema que localiza cada laranja numa imagem com uma caixa delimitadora, diz se ela está boa ou com defeito, informa a confiança da previsão e mostra um painel com a contagem e a taxa de defeito. Mais do que o número final, procurei documentar cada decisão, inclusive os erros do modelo, porque é neles que fica claro o que ele realmente aprendeu."),
  h2("1.1 Objetivo geral"),
  p("Desenvolver e treinar um modelo YOLO capaz de detectar laranjas e classificá-las como boas ou com defeito, avaliando seu desempenho em imagens que nunca participaram do treinamento."),
  h2("1.2 Objetivos específicos"),
  p("Os objetivos específicos seguem as etapas pedidas no enunciado da atividade:", { indent: { firstLine: RECUO } }),
  ...alineas([
    "escolher um problema de reconhecimento de objetos;",
    "obter um dataset de imagens e anotar os objetos no formato YOLO;",
    "preparar os dados, separando treino, validação e imagens inéditas;",
    "treinar um modelo YOLO leve com pesos pré-treinados;",
    "avaliar o desempenho com precision, recall e mAP50;",
    "aplicar o modelo em imagens novas e analisar acertos e erros;",
    "apresentar os resultados e explicar o funcionamento da aplicação.",
  ]),

  // ------------------------------------------------------------ 2
  h1("2 FUNDAMENTAÇÃO TEÓRICA"),
  h2("2.1 Detecção de objetos com YOLO"),
  p("Detectar um objeto é responder a duas perguntas ao mesmo tempo: onde ele está e o que ele é. A família YOLO (You Only Look Once) faz isso olhando a imagem uma única vez. A rede divide a imagem e prevê, de uma só vez, as caixas delimitadoras e as classes, o que a torna rápida o bastante para uso em tempo real (REDMON et al., 2016). Neste trabalho usei a versão YOLOv8n, da biblioteca Ultralytics, a menor da família, pensada para rodar em computadores comuns (JOCHER; CHAURASIA; QIU, 2023)."),
  h2("2.2 Transferência de aprendizado"),
  p("Treinar uma rede do zero exigiria milhares de imagens. Por isso parti de pesos pré-treinados no COCO, um conjunto com 80 classes de objetos do dia a dia (LIN et al., 2014). O modelo já sabe reconhecer bordas, texturas e formas. O treinamento apenas ajusta esse conhecimento para as duas classes deste projeto, e é isso que torna viável aprender com poucas centenas de imagens."),
  h2("2.3 Métricas de avaliação"),
  p("Para saber se uma caixa prevista acertou, compara-se a área de interseção com a caixa real dividida pela área da união das duas, o IoU. A partir disso, as métricas usadas foram:"),
  ...alineas([
    [{ t: "precision", bold: true }, ": das caixas que o modelo desenhou, quantas estavam certas. Mede os alarmes falsos;"],
    [{ t: "recall", bold: true }, ": das laranjas que existiam na imagem, quantas o modelo encontrou. Mede o que escapou;"],
    [{ t: "mAP50", bold: true }, ": nota de 0 a 1 que resume precision e recall em todos os níveis de confiança, contando como acerto a caixa com IoU de pelo menos 0,5;"],
    [{ t: "mAP50-95", bold: true }, ": a mesma nota com régua mais exigente, com média de IoU de 0,50 a 0,95, que mede se a caixa está bem encaixada."],
  ]),

  // ------------------------------------------------------------ 3
  h1("3 METODOLOGIA"),
  h2("3.1 Dataset e anotação"),
  p("Usei o Orange Fruit Diseases Dataset (DOH, 2025), publicado no Mendeley Data com licença CC BY 4.0. Ele reúne 700 fotos organizadas em pastas por diagnóstico: saudável (healthy), mancha preta (black spot), cancro cítrico (canker), greening e verrugose (scab). As fotos saudáveis formaram a classe laranja_boa, e as quatro doenças formaram a classe laranja_defeito."),
  p("O arquivo disponível trazia apenas as imagens, sem as caixas delimitadoras. Então fiz uma anotação semi-automática: o YOLOv8n pré-treinado no COCO propôs a caixa de cada fruta e, nas fotos de fundo liso em que ele falhou, uma segmentação simples por cor fez esse papel. A classe veio da pasta, ou seja, do diagnóstico dado pelos autores do dataset. Mantive só as fotos com uma única fruta, porque numa foto de árvore com várias laranjas o rótulo da pasta não vale para cada uma, removi 156 quase-duplicatas e conferi as caixas visualmente por amostragem. A composição final está na Tabela 1, e a Figura 1 mostra exemplos das anotações."),
  ...tabela("Composição do dataset por classe", ["Classe", "Treino", "Validação", "Inéditas", "Total"], [
    ["0: laranja_boa", "96", "24", "4", "124"],
    ["1: laranja_defeito", "96", "24", "4", "124"],
    ["Total", "192", "48", "8", "248"],
  ], [2900, 1450, 1450, 1450, 1450], `${AUTOR}, a partir de Doh (2025).`),
  ...figura("Exemplos de imagens de treino com as caixas anotadas", RES("dataset/exemplos_anotados.jpg"), 15, 1292, 864,
    `${AUTOR}, a partir de Doh (2025).`),
  h2("3.2 Preparação dos dados"),
  p("A divisão seguiu a recomendação do enunciado: cerca de 80% para treino e 20% para validação, com as imagens inéditas separadas antes de tudo. A divisão é estratificada, o que mantém a mesma proporção de classes dos dois lados, e usa uma semente fixa, para que o resultado seja reproduzível. Antes de dividir, o sistema valida cada linha de anotação no formato classe x_centro y_centro largura altura, com coordenadas normalizadas entre 0 e 1, e confere por hash que nenhuma imagem inédita é cópia de uma imagem de treino ou validação."),
  h2("3.3 Configuração do ambiente e do YOLO"),
  p("O ambiente foi Python 3.14 com Ultralytics 8.4.215 e PyTorch 2.15 em CPU, num processador Intel Core i5-13420H. A instalação foi conferida com o comando yolo checks. O arquivo data.yaml, gerado automaticamente, ficou assim:"),
  ...codigo(["path: <pasta do projeto>/backend/dataset", "train: images/train", "val: images/val", "names:", "  0: laranja_boa", "  1: laranja_defeito"]),
  h2("3.4 Treinamento"),
  p("O modelo YOLOv8n foi treinado por 30 épocas, com imagens de 640 pixels e lote (batch) de 4 imagens, na CPU. O treinamento completo levou cerca de 27 minutos. Ao final, a ferramenta salvou o arquivo best.pt com os melhores pesos segundo seu critério de avaliação."),
  h2("3.5 Validação e teste em imagens inéditas"),
  p("A validação foi feita com o best.pt sobre as 48 imagens de validação. Em seguida, o modelo foi aplicado às 8 imagens inéditas com confiança mínima de 0,25. Como essas imagens também têm gabarito, cada uma foi classificada automaticamente como acerto ou erro. O erro, por sua vez, foi separado em classe trocada, falso positivo ou fruta não detectada, considerando acerto apenas a caixa com IoU de pelo menos 0,5 e a classe correta."),
  h2("3.6 Arquitetura do sistema desenvolvido"),
  p("Para além dos comandos do enunciado, organizei o trabalho como um sistema. O backend concentra o pipeline em Python, as regras de qualidade, os testes e os resultados. O frontend reúne o que é apresentado: este documento, o notebook para o Google Colab e a documentação. Cada etapa é um comando que grava o que produziu, e a etapa seguinte lê desses arquivos. Por isso o relatório não tem nenhum número digitado à mão. O Quadro 1 resume as etapas."),
  ...quadro("Etapas do sistema e evidências geradas", ["Etapa", "Comando", "Evidência gerada"], [
    ["Preparo dos dados", "citrus-yolo preparar", "Divisão 80/20, data.yaml e checklist de qualidade"],
    ["Treinamento", "citrus-yolo treinar", "best.pt, curvas e tabela por época"],
    ["Validação", "citrus-yolo validar", "Precision, recall, mAP50 e mAP50-95 por classe"],
    ["Reconhecimento", "citrus-yolo detectar", "Painéis de inspeção e acerto/erro nas inéditas"],
    ["Relatório e entrega", "citrus-yolo relatorio / entregar", "Relatório PDF e arquivo ZIP"],
  ], [2300, 3000, 3771], AUTOR),
  p("O sistema também aplica regras de negócio automaticamente. Ele bloqueia anotações fora do formato, exige pelo menos 60 imagens anotadas, impede que uma imagem inédita seja cópia do treino, exige justificativa para menos de 20 épocas e registra cada comando executado com data e hora. Essas regras estão cobertas por 38 testes automatizados."),

  // ------------------------------------------------------------ 4
  h1("4 RESULTADOS E DISCUSSÃO"),
  h2("4.1 Treinamento"),
  p("A Tabela 2 mostra a evolução a cada cinco épocas. As perdas (losses) caíram de forma contínua, sinal de que o modelo estava aprendendo, e o mAP50 já passava de 0,97 na época 10. Daí em diante, o ganho veio sobretudo no mAP50-95, ou seja, no ajuste fino da posição das caixas. A Figura 2 mostra as curvas completas."),
  ...tabela("Evolução do treinamento", ["Época", "Box loss", "Cls loss", "Precision", "Recall", "mAP50", "mAP50-95"], [
    ["1", "0,722", "2,297", "0,510", "0,913", "0,783", "0,636"],
    ["5", "0,687", "0,962", "0,520", "0,891", "0,867", "0,683"],
    ["10", "0,603", "0,722", "0,912", "0,938", "0,975", "0,807"],
    ["15", "0,579", "0,629", "0,935", "0,955", "0,984", "0,825"],
    ["20", "0,503", "0,498", "0,963", "0,966", "0,982", "0,891"],
    ["25", "0,313", "0,379", "0,953", "0,961", "0,981", "0,898"],
    ["30", "0,276", "0,315", "0,959", "0,979", "0,986", "0,902"],
  ], [1100, 1300, 1300, 1300, 1300, 1300, 1300], `${AUTOR}, com dados do treinamento.`, AlignmentType.CENTER),
  ...figura("Curvas de perda e métricas ao longo das 30 épocas", RES("treino/results.png"), 16, 2400, 1200,
    `${AUTOR}, gerado pela biblioteca Ultralytics.`),
  p("O arquivo best.pt foi gerado em runs/detect/train/weights/best.pt, com 6,5 MB, e copiado para a pasta modelo/ da entrega. A época com maior mAP50-95 na validação foi a 27."),
  h2("4.2 Validação"),
  p("A Tabela 3 traz as métricas obtidas com o best.pt nas 48 imagens de validação."),
  ...tabela("Métricas de validação por classe", ["Classe", "Precision", "Recall", "mAP50", "mAP50-95"], [
    ["Todas", "0,899", "0,978", "0,986", "0,904"],
    ["laranja_boa", "0,839", "1,000", "0,990", "0,904"],
    ["laranja_defeito", "0,958", "0,957", "0,982", "0,904"],
  ], [2671, 1600, 1600, 1600, 1600], `${AUTOR}, com dados da validação.`),
  p("O recall de 0,978 mostra que quase nenhuma laranja passou sem ser detectada. O mAP50-95 de 0,904 indica caixas bem ajustadas à fruta. O ponto de atenção é a precision da classe laranja_boa, de 0,839: parte das caixas marcadas como boas estava errada. Numa linha de beneficiamento, esse é o erro mais caro, porque significa uma fruta doente seguindo para o cliente. A matriz de confusão da Figura 3 ajuda a ver onde estão as trocas."),
  ...figura("Matriz de confusão normalizada da validação", RES("validacao/confusion_matrix_normalized.png"), 12, 3000, 2250,
    `${AUTOR}, gerado pela biblioteca Ultralytics.`),
  h2("4.3 Reconhecimento em imagens inéditas"),
  p("A Tabela 4 resume o comportamento do modelo nas 8 imagens que ele nunca tinha visto. Ele localizou as frutas, informou a classe e a confiança e acertou 5 imagens sem nenhum erro."),
  ...tabela("Resultado nas imagens inéditas", ["Imagem", "Detecção (classe e confiança)", "Resultado"], [
    ["Black spot 117", "laranja_defeito 0,95", "Acerto"],
    ["Canker 45", "laranja_boa 0,67 e laranja_defeito 0,55", "Erro: classe trocada"],
    ["Greening 59", "laranja_defeito 0,94", "Acerto"],
    ["Healthy 171", "laranja_boa 0,94", "Acerto"],
    ["Healthy 195", "laranja_boa 0,91", "Acerto"],
    ["Healthy 59", "laranja_boa 0,87", "Acerto"],
    ["Healthy 61", "nenhuma", "Erro: não detectada"],
    ["Scab 109", "laranja_boa 0,75", "Erro: classe trocada"],
  ], [2300, 4200, 2571], `${AUTOR}, com dados da detecção.`),
  ...figura("Painéis de inspeção gerados nas imagens inéditas", RES("deteccoes/mosaico_deteccoes.jpg"), 11, 1144, 1540, AUTOR),
  p("Os três erros dizem mais do que os acertos. Na foto de cancro, o modelo desenhou duas caixas na mesma fruta, uma de boa e outra de defeito, ambas com confiança abaixo de 0,70: ele ficou em dúvida, e o cancro tinha apenas 26 exemplos somando treino e validação. Na foto de verrugose, a fruta foi classificada como boa com confiança de 0,75, e essa doença quase não entrou no dataset, com só 4 exemplos. Na laranja saudável não detectada, a foto estava girada 45 graus, com uma folha seca por cima e a fruta cortada nas bordas, uma situação diferente de tudo o que o modelo viu no treino."),
  p("Há ainda uma diferença importante: a validação deu mAP50 de 0,986, mas nas imagens inéditas o acerto foi de 5 em 8. A explicação mais provável é que o dataset público tem cópias giradas da mesma foto, que a remoção de duplicatas não detecta, e assim versões de uma mesma imagem podem ter caído no treino e na validação. Por isso considero as imagens inéditas o retrato mais fiel do desempenho real."),

  // ------------------------------------------------------------ 5
  h1("5 RESPOSTAS ÀS QUESTÕES DA ATIVIDADE"),
  h2("5.1 Questão 1: problema escolhido e motivo"),
  p("Escolhi detectar laranjas e separá-las em boas e com defeito, porque é um problema real do beneficiamento de citros. A triagem manual é cansativa e varia de pessoa para pessoa, e a diferença entre fruta sadia e doente aparece na imagem, o que combina com um modelo pequeno treinado com poucas centenas de fotos."),
  h2("5.2 Questão 2: imagens, classes e origem"),
  p("Foram usadas 248 imagens: 192 de treino, 48 de validação e 8 inéditas. As classes são 0: laranja_boa e 1: laranja_defeito, esta última reunindo mancha preta, cancro, greening e verrugose. As imagens vieram do Orange Fruit Diseases Dataset (DOH, 2025) e foram anotadas de forma semi-automática, como descrito na seção 3.1."),
  h2("5.3 Questão 3: modelo e épocas"),
  p("Usei o YOLOv8n, com pesos pré-treinados no COCO, treinado por 30 épocas, com imagens de 640 pixels e lote de 4, em CPU."),
  h2("5.4 Questão 4: métricas e significado"),
  p("Na validação, obtive precision de 0,899, recall de 0,978 e mAP50 de 0,986, além de mAP50-95 de 0,904. A precision diz se o modelo acerta quando aponta, o recall diz se ele encontra o que existe e o mAP50 resume os dois numa nota única, como explicado na seção 2.3. Os valores são altos, mas, como discutido na seção 4.3, estão otimistas."),
  h2("5.5 Questão 5: reconhecimento nas imagens inéditas"),
  p("Sim, na maior parte dos casos. O modelo acertou 5 de 8 imagens sem nenhum erro, com confiança de 0,87 ou mais nos acertos. Os erros aconteceram em doenças com poucos exemplos no treino e numa foto atípica, como mostram a Tabela 4 e a Figura 4."),
  h2("5.6 Questão 6: dificuldades e melhorias"),
  p("A maior dificuldade foi o próprio dataset: as anotações não vinham no arquivo, muitas fotos tinham várias frutas e havia muitas cópias da mesma imagem. Também notei que as fotos saudáveis eram quase todas de estúdio, com fundo branco, enquanto as doentes tinham fundos mais escuros, o que pode levar o modelo a usar o fundo como pista. Para melhorar, eu equilibraria as doenças dentro da classe defeito, separaria treino e validação pela foto de origem, fotografaria laranjas reais em fundos e luzes variados, revisaria as anotações à mão e, com uma GPU, testaria um modelo maior por mais épocas."),

  // ------------------------------------------------------------ 6
  h1("6 CONCLUSÃO"),
  p("O trabalho cumpriu todos os requisitos da atividade: duas classes, 240 imagens anotadas para treino e validação, 30 épocas de treinamento, validação com as métricas pedidas e teste em 8 imagens inéditas. O modelo aprendeu a localizar laranjas com muita segurança e, na maior parte dos casos, a diferenciar fruta boa de fruta doente."),
  p("A lição mais valiosa, porém, não está no mAP alto. Está na distância entre a validação e as imagens inéditas. Ela mostrou, na prática, que a qualidade dos dados pesa tanto quanto o modelo: duplicatas escondidas, classes mal representadas e fundos diferentes entre as classes mudam o que a rede aprende. Entendi a diferença entre usar um modelo pronto e adaptar um modelo a um problema novo, e também por que avaliar em dados nunca vistos é indispensável antes de confiar em qualquer número."),
  p("Como continuidade, o projeto pode evoluir para um piloto com imagens da própria esteira de uma unidade de beneficiamento, com anotação manual por tipo de doença e câmera em tempo real."),

  // ------------------------------------------------------------ pós-textuais
  h1SemNumero("REFERÊNCIAS"),
  ...[
    [{ t: "DOH, B. " }, { t: "Orange Fruit Diseases Dataset", bold: true }, { t: ". Versão 1. [S. l.]: Mendeley Data, 2025. DOI: 10.17632/6szsnpypdd.1. Disponível em: https://data.mendeley.com/datasets/6szsnpypdd. Acesso em: 17 nov. 2026." }],
    [{ t: "JOCHER, G.; CHAURASIA, A.; QIU, J. " }, { t: "Ultralytics YOLOv8", bold: true }, { t: ". Versão 8.0.0. [S. l.]: Ultralytics, 2023. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 17 nov. 2026." }],
    [{ t: "LIN, T.-Y. et al. Microsoft COCO: common objects in context. In: EUROPEAN CONFERENCE ON COMPUTER VISION, 13., 2014, Zurique. " }, { t: "Proceedings", bold: true }, { t: " [...]. Cham: Springer, 2014. p. 740-755." }],
    [{ t: "REDMON, J.; DIVVALA, S.; GIRSHICK, R.; FARHADI, A. You Only Look Once: unified, real-time object detection. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION, 2016, Las Vegas. " }, { t: "Proceedings", bold: true }, { t: " [...]. Piscataway: IEEE, 2016. p. 779-788." }],
  ].map((ref) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { line: 240, after: 240 }, children: partes(ref) })),

  h1SemNumero("APÊNDICE A – COMANDOS UTILIZADOS"),
  p("Comandos executados de dentro da pasta backend/. A lista completa, com data e hora de cada execução, está no arquivo comandos.txt da entrega.", { indent: { firstLine: 0 } }),
  ...codigo([
    "yolo checks",
    "python scripts/anotar_mendeley.py \"<pasta>/Orange Fruit Diseases\"",
    "citrus-yolo preparar",
    "citrus-yolo treinar",
    "citrus-yolo validar",
    "citrus-yolo detectar",
    "citrus-yolo relatorio",
    "citrus-yolo entregar",
    "",
    "Equivalentes na CLI do Ultralytics:",
    "yolo detect train model=yolov8n.pt data=\"dataset/data.yaml\" epochs=30 imgsz=640 batch=4",
    "yolo detect val model=\"modelo/best.pt\" data=\"dataset/data.yaml\"",
    "yolo detect predict model=\"modelo/best.pt\" source=\"dataset/ineditas\" conf=0.25 save=True",
  ]),

  h1SemNumero("APÊNDICE B – ESTRUTURA DA ENTREGA E USO DE IA"),
  p("O arquivo ZIP da entrega segue a estrutura sugerida no enunciado:", { indent: { firstLine: 0 } }),
  ...codigo([
    "Projeto_Final_YOLO_Bruno_Henrique_da_Silveira/",
    "├── dataset/        imagens, anotações, inéditas e data.yaml",
    "├── modelo/best.pt  pesos treinados",
    "├── resultados/     treino, validação e detecções",
    "├── relatorio.pdf   relatório com as seis questões",
    "├── comandos.txt    comandos efetivamente utilizados",
    "└── codigo/         backend/ e frontend/ do sistema",
  ]),
  new Paragraph({ spacing: { before: 240 }, children: [] }),
  p("Por transparência, registro que usei assistentes de inteligência artificial (Claude Code, no VS Code) como apoio de programação: na estrutura do projeto, no código e nos testes, na busca do dataset, no script de anotação semi-automática, na execução do treino e no rascunho dos textos a partir dos resultados reais. A escolha do tema, a conferência dos dados e das anotações, a revisão dos textos e a análise final são de minha responsabilidade. Nenhum número deste documento foi digitado à mão: todos vêm dos arquivos gerados pelo sistema. O registro completo está em frontend/docs/uso_de_llms.md."),
];

// ---------------------------------------------------------------- documento
const cabecalhoNumero = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONTE, size: T10 })] })] });
const pagina = { size: { width: 11906, height: 16838 }, margin: MARGENS };

const doc = new Document({
  creator: "Bruno Henrique da Silveira",
  title: "Atividade Prática Final – Inteligência Artificial: modelo de reconhecimento de objetos com YOLO",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONTE, size: T12 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONTE, size: T12, bold: true, allCaps: true },
        paragraph: { spacing: { before: 0, after: LINHA_15, line: LINHA_15 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONTE, size: T12, bold: true },
        paragraph: { spacing: { before: LINHA_15, after: LINHA_15, line: LINHA_15 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "alineas", levels: [{ level: 0, format: LevelFormat.LOWER_LETTER, text: "%1)",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: RECUO + 360, hanging: 360 } } } }] }] },
  sections: [
    { properties: { page: pagina }, children: capa },
    // folha de rosto conta como página 1 (NBR 14724), mas o número só aparece a partir da introdução
    { properties: { page: { ...pagina, pageNumbers: { start: 1 } } }, children: [...folhaDeRosto, ...resumo, ...sumario] },
    { properties: { page: pagina }, headers: { default: cabecalhoNumero }, children: textual },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.argv[2], buf); console.log("ok", buf.length); });
