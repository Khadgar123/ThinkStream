const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.author = "ThinkStream Team";
pres.title = "ThinkStream: Thinking in Streaming Video";

// ─── Color Palette ───
const C = {
  midnight: "21295C",
  deep: "065A82",
  teal: "1C7293",
  light_teal: "2A9DB5",
  ice: "CADCFC",
  white: "FFFFFF",
  off_white: "F4F7FA",
  light_gray: "E8EDF2",
  dark_text: "1A1A2E",
  sub_text: "4A5568",
  accent_red: "E53E3E",
  accent_green: "38A169",
  accent_orange: "DD6B20",
  accent_purple: "805AD5",
};

// ─── Helpers ───
const mkShadow = () => ({ type: "outer", color: "000000", blur: 4, offset: 2, angle: 135, opacity: 0.12 });
const mkCardShadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 3, angle: 135, opacity: 0.15 });

function addSlideNumber(slide, num, total) {
  slide.addText(`${num} / ${total}`, {
    x: 8.8, y: 5.25, w: 1, h: 0.3,
    fontSize: 9, color: C.sub_text, align: "right", fontFace: "Calibri",
  });
}

function addSectionTitle(slide, title, opts = {}) {
  slide.addText(title, {
    x: opts.x || 0.6, y: opts.y || 0.25, w: opts.w || 8.8, h: 0.55,
    fontSize: opts.fontSize || 26, fontFace: "Georgia", color: C.midnight, bold: true, margin: 0,
  });
  // Accent bar under title
  slide.addShape(pres.shapes.RECTANGLE, {
    x: opts.x || 0.6, y: (opts.y || 0.25) + 0.55, w: 1.2, h: 0.04,
    fill: { color: C.teal },
  });
}

function addCard(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.white },
    shadow: mkCardShadow(),
    line: opts.border ? { color: opts.border, width: 1.5 } : undefined,
  });
}

function addIconCircle(slide, x, y, size, color, text) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: size, h: size,
    fill: { color },
  });
  slide.addText(text, {
    x, y, w: size, h: size,
    fontSize: size * 14, fontFace: "Calibri", color: C.white,
    align: "center", valign: "middle", bold: true, margin: 0,
  });
}

function addArrowRight(slide, x, y, w) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h: 0,
    line: { color: C.teal, width: 2 },
  });
  // Arrowhead triangle
  slide.addText("\u25B6", {
    x: x + w - 0.15, y: y - 0.12, w: 0.25, h: 0.25,
    fontSize: 10, color: C.teal, align: "center", valign: "middle", margin: 0,
  });
}

function addArrowDown(slide, x, y, h) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w: 0, h,
    line: { color: C.teal, width: 2 },
  });
  slide.addText("\u25BC", {
    x: x - 0.1, y: y + h - 0.1, w: 0.2, h: 0.2,
    fontSize: 8, color: C.teal, align: "center", valign: "middle", margin: 0,
  });
}

const TOTAL = 16;

// ════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.midnight };

  // Decorative shapes
  s.addShape(pres.shapes.OVAL, { x: 7.5, y: -1.0, w: 4.5, h: 4.5, fill: { color: C.deep, transparency: 60 } });
  s.addShape(pres.shapes.OVAL, { x: 8.5, y: 2.5, w: 3.5, h: 3.5, fill: { color: C.teal, transparency: 70 } });
  s.addShape(pres.shapes.OVAL, { x: 6.0, y: 3.0, w: 2.0, h: 2.0, fill: { color: C.light_teal, transparency: 75 } });

  // Left accent bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.2, w: 0.06, h: 2.5, fill: { color: C.light_teal } });

  s.addText("ThinkStream", {
    x: 0.8, y: 1.2, w: 7, h: 0.8,
    fontSize: 44, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  s.addText("Thinking in Streaming Video", {
    x: 0.8, y: 1.95, w: 7, h: 0.55,
    fontSize: 24, fontFace: "Georgia", color: C.ice, italic: true, margin: 0,
  });
  s.addText([
    { text: "流式视频理解", options: { fontSize: 18, color: C.ice } },
    { text: "  \u00B7  ", options: { fontSize: 18, color: C.light_teal } },
    { text: "边看边思考", options: { fontSize: 18, color: C.ice } },
    { text: "  \u00B7  ", options: { fontSize: 18, color: C.light_teal } },
    { text: "无限长度", options: { fontSize: 18, color: C.ice } },
  ], { x: 0.8, y: 2.65, w: 7, h: 0.45, fontFace: "Calibri", margin: 0 });

  // Bottom info
  s.addText("arXiv: 2603.12938  |  2026", {
    x: 0.8, y: 4.7, w: 5, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.sub_text,
  });

  // Streaming wave decoration
  for (let i = 0; i < 5; i++) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.0 + i * 0.25, y: 1.0 + Math.sin(i * 0.8) * 0.3, w: 0.12, h: 0.6 + Math.sin(i * 1.2) * 0.3,
      fill: { color: C.ice, transparency: 40 + i * 8 },
    });
  }

  addSlideNumber(s, 1, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem Statement
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Why Streaming? \u4E3A\u4EC0\u4E48\u9700\u8981\u6D41\u5F0F\u7406\u89E3");

  // Left column — Problems
  addCard(s, 0.4, 1.1, 4.3, 3.8, { border: C.accent_red });
  s.addText("\u4F20\u7EDF\u65B9\u6CD5\u7684\u74F6\u9888", {
    x: 0.6, y: 1.2, w: 3.9, h: 0.4,
    fontSize: 16, fontFace: "Calibri", color: C.accent_red, bold: true, margin: 0,
  });

  const problems = [
    ["\u23F0", "\u5FC5\u987B\u7B49\u89C6\u9891\u64AD\u653E\u5B8C\u624D\u80FD\u56DE\u7B54", "\u65E0\u6CD5\u5B9E\u65F6\u4EA4\u4E92\uFF0C\u54CD\u5E94\u5EF6\u8FDF\u6781\u9AD8"],
    ["\uD83D\uDCBE", "1\u5C0F\u65F6\u89C6\u9891 \u2192 ~1M tokens \u2192 57GB VRAM", "\u663E\u5B58\u6D88\u8017\u4E0E\u89C6\u9891\u957F\u5EA6\u7EBF\u6027\u589E\u957F"],
    ["\uD83D\uDD12", "\u65E0\u6CD5\u5904\u7406\u65E0\u9650\u957F\u76F4\u64AD/\u76D1\u63A7\u573A\u666F", "\u5185\u5B58\u7A97\u53E3\u56FA\u5B9A\uFF0C\u53EA\u80FD\u770B\u6709\u9650\u7247\u6BB5"],
  ];
  problems.forEach((p, i) => {
    const py = 1.75 + i * 1.0;
    addIconCircle(s, 0.65, py, 0.4, C.accent_red, p[0]);
    s.addText(p[1], { x: 1.2, y: py, w: 3.3, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.dark_text, bold: true, margin: 0 });
    s.addText(p[2], { x: 1.2, y: py + 0.3, w: 3.3, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.sub_text, margin: 0 });
  });

  // Right column — Solutions
  addCard(s, 5.3, 1.1, 4.3, 3.8, { border: C.accent_green });
  s.addText("ThinkStream \u7684\u7A81\u7834", {
    x: 5.5, y: 1.2, w: 3.9, h: 0.4,
    fontSize: 16, fontFace: "Calibri", color: C.accent_green, bold: true, margin: 0,
  });

  const solutions = [
    ["\u26A1", "\u6BCF2\u79D2\u5B9E\u65F6\u63A8\u7406\uFF0C\u5373\u65F6\u54CD\u5E94", "\u7528\u6237\u63D0\u95EE\u540E\u53EF\u5728\u5F53\u524D\u6B65\u7ACB\u5373\u56DE\u7B54"],
    ["\uD83D\uDDDC", "\u56FA\u5B9A ~3,400 tokens\uFF0C\u4E0E\u89C6\u9891\u957F\u5EA6\u65E0\u5173", "\u538B\u7F29\u8BB0\u5FC6\u673A\u5236\u4FDD\u8BC1\u6052\u5B9A\u5185\u5B58\u5360\u7528"],
    ["\u221E", "\u7406\u8BBA\u652F\u6301\u65E0\u9650\u957F\u5EA6\u89C6\u9891", "\u6EDA\u52A8\u7A97\u53E3 + \u538B\u7F29\u5F52\u6863\uFF0C\u65E0\u4E0A\u9650"],
  ];
  solutions.forEach((p, i) => {
    const py = 1.75 + i * 1.0;
    addIconCircle(s, 5.55, py, 0.4, C.accent_green, p[0]);
    s.addText(p[1], { x: 6.1, y: py, w: 3.3, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.dark_text, bold: true, margin: 0 });
    s.addText(p[2], { x: 6.1, y: py + 0.3, w: 3.3, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.sub_text, margin: 0 });
  });

  // Bottom stat
  addCard(s, 2.5, 5.05, 5.0, 0.45);
  s.addText([
    { text: "90% ", options: { fontSize: 18, bold: true, color: C.teal } },
    { text: "KV prefill reduction per step", options: { fontSize: 13, color: C.dark_text } },
  ], { x: 2.5, y: 5.05, w: 5.0, h: 0.45, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });

  addSlideNumber(s, 2, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 3 — Watch-Think-Speak Architecture
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Watch-Think-Speak \u4E09\u9636\u6BB5\u67B6\u6784");

  // Three phase boxes
  const phases = [
    { label: "WATCH \u89C2\u5BDF", color: C.deep, desc: "2\u79D2\u89C6\u9891\u5757\n24\u5E27\u6ED1\u52A8\u7A97\u53E3\n\u63A5\u6536\u65B0\u7684\u89C6\u89C9\u8F93\u5165" },
    { label: "THINK \u601D\u8003", color: C.teal, desc: "40-60 tokens\n\u589E\u91CF\u5F0F\u63A8\u7406\n\u751F\u6210\u89C2\u5BDF\u8BB0\u5FC6" },
    { label: "SPEAK \u884C\u52A8", color: C.light_teal, desc: "4\u79CD\u52A8\u4F5C\u9009\u62E9\nsilent / response\nrecall / compress" },
  ];

  phases.forEach((p, i) => {
    const px = 0.6 + i * 3.2;
    addCard(s, px, 1.15, 2.6, 2.2);
    // Phase header
    s.addShape(pres.shapes.RECTANGLE, {
      x: px, y: 1.15, w: 2.6, h: 0.5,
      fill: { color: p.color },
    });
    s.addText(p.label, {
      x: px, y: 1.15, w: 2.6, h: 0.5,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
    });
    s.addText(p.desc, {
      x: px + 0.15, y: 1.75, w: 2.3, h: 1.5,
      fontSize: 12, fontFace: "Calibri", color: C.dark_text, valign: "top", margin: 0, lineSpacingMultiple: 1.3,
    });

    // Arrows between boxes
    if (i < 2) {
      addArrowRight(s, px + 2.6, 2.25, 0.6);
    }
  });

  // Timeline bar below
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 3.65, w: 8.8, h: 0.04, fill: { color: C.teal },
  });
  const times = ["t=0", "t=2", "t=4", "t=6", "t=8", "t=10", "...", "t=\u221E"];
  times.forEach((t, i) => {
    const tx = 0.6 + i * 1.15;
    if (t !== "...") {
      s.addShape(pres.shapes.OVAL, { x: tx, y: 3.57, w: 0.18, h: 0.18, fill: { color: C.teal } });
    }
    s.addText(t, { x: tx - 0.2, y: 3.8, w: 0.6, h: 0.25, fontSize: 9, fontFace: "Calibri", color: C.sub_text, align: "center", margin: 0 });
  });
  s.addText("\u8FDE\u7EED\u6D41\u5F0F\u5904\u7406\uFF0C\u6BCF2\u79D2\u4E00\u4E2A\u5FAA\u73AF", {
    x: 0.6, y: 4.1, w: 4, h: 0.25, fontSize: 10, fontFace: "Calibri", color: C.sub_text, italic: true, margin: 0,
  });

  // Key point callout
  addCard(s, 0.6, 4.5, 9.0, 0.85, { border: C.teal });
  s.addText([
    { text: "\u6838\u5FC3\u8BBE\u8BA1\uFF1A", options: { bold: true, color: C.teal, fontSize: 13 } },
    { text: "\u6BCF\u4E2A timestep \u72EC\u7ACB\u91CD\u65B0\u6E32\u67D3 (Per-Timestep Re-render)\uFF0C\u4E0D\u7D2F\u79EF KV cache\u3002\u8F93\u5165 = [system] + [\u89C6\u89C9\u7A97\u53E3] + [\u8BB0\u5FC6\u6587\u672C] + [\u7528\u6237\u8F93\u5165]\uFF0C\u603B\u957F ~3,400 tokens\u3002H100 \u4E0A\u6BCF\u6B65 ~200ms\uFF0C\u8FDC\u5C0F\u4E8E 2\u79D2\u5B9E\u65F6\u9884\u7B97\u3002", options: { color: C.dark_text, fontSize: 12 } },
  ], { x: 0.8, y: 4.55, w: 8.6, h: 0.75, fontFace: "Calibri", valign: "middle", margin: 0, lineSpacingMultiple: 1.3 });

  addSlideNumber(s, 3, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 4 — Memory Architecture (RCSM)
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Reasoning-Compressed Streaming Memory (RCSM)");

  // Left: Memory stack diagram
  const layers = [
    { label: "Compressed Segments \u538B\u7F29\u6BB5", desc: "[0-20s] Chef prepared workspace...", tok: "\u2264750 tok", color: C.deep, max: "max 5 segments" },
    { label: "Recent Thinks \u8FD1\u671F\u601D\u8003", desc: "[40-42] Oil heated... [42-44] Garlic added...", tok: "\u2264600 tok", color: C.teal, max: "max 12 items" },
    { label: "Pending Questions \u5F85\u7B54\u95EE\u9898", desc: '{"since":44, "question":"when basil?"}', tok: "variable", color: C.light_teal, max: "\u4E8B\u4EF6\u76D1\u63A7" },
    { label: "Retrieval Archive \u68C0\u7D22\u5F52\u6863", desc: "\u5386\u53F2\u5B58\u50A8\uFF0C\u4F9B recall \u68C0\u7D22 (\u6A21\u578B\u4E0D\u53EF\u89C1)", tok: "system", color: C.sub_text, max: "hidden from model" },
  ];

  layers.forEach((l, i) => {
    const ly = 1.1 + i * 0.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y: ly, w: 5.6, h: 0.82,
      fill: { color: C.white },
      line: { color: l.color, width: 1.5 },
      shadow: mkShadow(),
    });
    // Color accent left
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: ly, w: 0.08, h: 0.82, fill: { color: l.color } });
    s.addText(l.label, { x: 0.65, y: ly + 0.02, w: 3.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: l.color, bold: true, margin: 0 });
    s.addText(l.desc, { x: 0.65, y: ly + 0.32, w: 4.0, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.sub_text, margin: 0 });
    s.addText(l.tok, { x: 4.5, y: ly + 0.02, w: 1.3, h: 0.3, fontSize: 11, fontFace: "Calibri", color: l.color, bold: true, align: "right", margin: 0 });
    s.addText(l.max, { x: 4.5, y: ly + 0.32, w: 1.3, h: 0.3, fontSize: 9, fontFace: "Calibri", color: C.sub_text, align: "right", margin: 0 });
  });

  // Right: Token budget pie chart
  s.addChart(pres.charts.DOUGHNUT, [{
    name: "Token Budget",
    labels: ["System 150", "Visual 1536", "Compressed 750", "Recent 600", "User 100"],
    values: [150, 1536, 750, 600, 100],
  }], {
    x: 6.2, y: 1.0, w: 3.5, h: 2.8,
    showPercent: true,
    showTitle: true,
    title: "Token Budget (~3,400)",
    titleColor: C.dark_text,
    titleFontSize: 11,
    chartColors: [C.sub_text, C.deep, C.teal, C.light_teal, C.ice],
    legendPos: "b",
    legendFontSize: 8,
  });

  // Key insight
  addCard(s, 6.2, 4.0, 3.5, 0.8, { border: C.teal });
  s.addText([
    { text: "\u6838\u5FC3\u6D1E\u5BDF\uFF1A", options: { bold: true, color: C.teal, fontSize: 11 } },
    { text: "\u6587\u672C\u8BB0\u5FC6\u7684\u65F6\u95F4\u8DE8\u5EA6 > \u89C6\u89C9\u7A97\u53E3\uFF0C\u538B\u7F29\u6BB5\u5F52\u7EB3\u4E86\u66F4\u4E45\u8FDC\u7684\u8FC7\u53BB\uFF0C\u652F\u6301\u957F\u7A0B\u4F9D\u8D56", options: { color: C.dark_text, fontSize: 11 } },
  ], { x: 6.35, y: 4.05, w: 3.2, h: 0.7, fontFace: "Calibri", valign: "middle", margin: 0, lineSpacingMultiple: 1.3 });

  // Bottom note
  s.addText("\u56FA\u5B9A\u5185\u5B58\u5360\u7528 \u2192 1\u5206\u949F\u89C6\u9891\u548C1\u5C0F\u65F6\u89C6\u9891\u6D88\u8017\u76F8\u540C\u7684 token \u9884\u7B97\uFF0C\u771F\u6B63\u5B9E\u73B0\u65E0\u9650\u957F\u5EA6", {
    x: 0.4, y: 5.0, w: 5.6, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: C.sub_text, italic: true, margin: 0,
  });

  addSlideNumber(s, 4, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 5 — Long-Range Dependency Timeline
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u957F\u7A0B\u4F9D\u8D56\uFF1A\u6587\u672C\u8BB0\u5FC6\u8986\u76D6\u66F4\u8FDC\u7684\u8FC7\u53BB");

  // Timeline base
  const tlY = 1.8;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: tlY + 0.9, w: 8.8, h: 0.03, fill: { color: C.sub_text } });

  // Time markers
  const markers = [
    { t: "t=0", x: 0.6 }, { t: "t=20", x: 2.8 }, { t: "t=40", x: 5.0 },
    { t: "t=46", x: 5.7 }, { t: "t=70 (now)", x: 9.0 },
  ];
  markers.forEach(m => {
    s.addShape(pres.shapes.RECTANGLE, { x: m.x, y: tlY + 0.8, w: 0.02, h: 0.22, fill: { color: C.sub_text } });
    s.addText(m.t, { x: m.x - 0.3, y: tlY + 1.05, w: 0.7, h: 0.25, fontSize: 9, fontFace: "Calibri", color: C.sub_text, align: "center", margin: 0 });
  });

  // Compressed zone
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: tlY + 0.1, w: 4.4, h: 0.65, fill: { color: C.deep, transparency: 20 }, line: { color: C.deep, width: 1 } });
  s.addText("Compressed Segments\n\u538B\u7F29\u6BB5: \u9AD8\u5C42\u8BED\u4E49\u6982\u62EC", {
    x: 0.7, y: tlY + 0.12, w: 4.2, h: 0.6, fontSize: 10, fontFace: "Calibri", color: C.deep, valign: "middle", margin: 0,
  });

  // Recent thinks zone
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: tlY + 0.1, w: 4.4, h: 0.65, fill: { color: C.teal, transparency: 20 }, line: { color: C.teal, width: 1 } });
  s.addText("Recent Thinks\n\u8FD1\u671F\u601D\u8003: \u8BE6\u7EC6\u9010\u6B65\u89C2\u5BDF", {
    x: 5.1, y: tlY + 0.12, w: 4.2, h: 0.6, fontSize: 10, fontFace: "Calibri", color: C.teal, valign: "middle", margin: 0,
  });

  // Visual window (overlay below)
  s.addShape(pres.shapes.RECTANGLE, { x: 5.7, y: tlY + 1.4, w: 3.7, h: 0.45, fill: { color: C.light_teal, transparency: 25 }, line: { color: C.light_teal, width: 1 } });
  s.addText("Visual Window [46-70s] 24 \u5E27\u539F\u59CB\u89C6\u9891", {
    x: 5.8, y: tlY + 1.42, w: 3.5, h: 0.4, fontSize: 10, fontFace: "Calibri", color: C.light_teal, valign: "middle", margin: 0,
  });

  // Text memory span bracket
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: tlY + 2.0, w: 8.8, h: 0.03, fill: { color: C.accent_orange } });
  s.addText("\u6587\u672C\u8BB0\u5FC6\u8DE8\u5EA6 (Text Memory Span) \u2014 \u8986\u76D6\u5168\u90E8\u65F6\u95F4\u8303\u56F4", {
    x: 2.0, y: tlY + 2.05, w: 6.0, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.accent_orange, bold: true, align: "center", margin: 0,
  });

  // Three explanation cards
  const cards = [
    { title: "Recall \u80FD\u529B", desc: "t=28\u7684\u8BC1\u636E\u5DF2\u79BB\u5F00\u89C6\u89C9\u7A97\u53E3\uFF0C\n\u4F46\u4ECD\u5728 archive \u4E2D\u53EF\u68C0\u7D22\uFF0C\n\u901A\u8FC7 recall \u673A\u5236\u627E\u56DE", color: C.deep },
    { title: "\u538B\u7F29\u6548\u7387", desc: "\u65E7\u7684\u8BB0\u5FC6\u4FDD\u7559\u9AD8\u5C42\u6982\u8981\uFF0C\n\u8FD1\u671F\u8BB0\u5FC6\u4FDD\u7559\u5B8C\u6574\u7EC6\u8282\uFF0C\n\u5F62\u6210\u591A\u5C42\u6B21\u4FE1\u606F\u5BC6\u5EA6", color: C.teal },
    { title: "\u957F\u7A0B\u4E0A\u4E0B\u6587", desc: "t=70\u7684\u56DE\u7B54\u53EF\u4EE5\u5F15\u7528\nt=0\u7684\u538B\u7F29\u6458\u8981\u4FE1\u606F\uFF0C\n\u7406\u8BBA\u4E0A\u65E0\u65F6\u95F4\u8DDD\u79BB\u9650\u5236", color: C.light_teal },
  ];
  cards.forEach((c, i) => {
    const cx = 0.5 + i * 3.2;
    addCard(s, cx, 3.8, 2.8, 1.55, { border: c.color });
    s.addText(c.title, { x: cx + 0.15, y: 3.85, w: 2.5, h: 0.35, fontSize: 13, fontFace: "Calibri", color: c.color, bold: true, margin: 0 });
    s.addText(c.desc, { x: cx + 0.15, y: 4.2, w: 2.5, h: 1.0, fontSize: 10, fontFace: "Calibri", color: C.dark_text, margin: 0, lineSpacingMultiple: 1.3 });
  });

  addSlideNumber(s, 5, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 6 — Compression Mechanism C1 vs C2
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u538B\u7F29\u673A\u5236\uFF1AC1 \u7CFB\u7EDF\u6307\u5BFC \u2192 C2 \u81EA\u4E3B\u9009\u62E9");

  // Trigger condition bar
  addCard(s, 0.5, 1.0, 9.0, 0.55);
  s.addText([
    { text: "\u89E6\u53D1\u6761\u4EF6\uFF1A", options: { bold: true, color: C.teal, fontSize: 12 } },
    { text: "recent_thinks tokens \u2265 80% budget (480 tok) \u2192 \u89E6\u53D1\u538B\u7F29 | \u538B\u7F29\u540E\u964D\u81F3 \u226455% (330 tok) \u2192 \u9632\u6B62\u9707\u8361 (hysteresis)", options: { color: C.dark_text, fontSize: 11 } },
  ], { x: 0.7, y: 1.05, w: 8.6, h: 0.45, fontFace: "Calibri", valign: "middle", margin: 0 });

  // Left: C1
  addCard(s, 0.4, 1.75, 4.4, 3.4, { border: C.deep });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.75, w: 4.4, h: 0.45, fill: { color: C.deep } });
  s.addText("Phase C1: Teacher-Guided \u6559\u5E08\u6307\u5BFC", {
    x: 0.5, y: 1.75, w: 4.2, h: 0.45, fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0,
  });

  // C1 flow
  const c1Steps = [
    "\u2460 \u591A\u7EF4\u5EA6\u8BC4\u5206\uFF1A\u679A\u4E3E\u6240\u6709\u5408\u6CD5 4-8 think \u8303\u56F4",
    "\u2461 \u8BC4\u5206\u7EF4\u5EA6\uFF1A",
    "   \u2022 importance_lost (\u5B9E\u4F53/OCR/\u6570\u5B57\u635F\u5931)",
    "   \u2022 pending_overlap_penalty (\u5F85\u7B54\u4FE1\u606F\u4FDD\u62A4)",
    "   \u2022 event_boundary_penalty (\u4E8B\u4EF6\u8FB9\u754C\u4FDD\u62A4)",
    "   \u2022 token_saving_gain (\u8282\u7701\u5956\u52B1)",
    "   \u2022 reconstructability_bonus (\u53EF\u91CD\u5EFA\u5956\u52B1)",
    "\u2462 \u9009\u62E9\u6700\u4F18\u8303\u56F4 \u2192 \u751F\u6210\u6458\u8981",
  ];
  s.addText(c1Steps.map((t, i) => ({
    text: t, options: { breakLine: i < c1Steps.length - 1, fontSize: t.startsWith("   ") ? 10 : 11, color: t.startsWith("   ") ? C.sub_text : C.dark_text, bold: t.startsWith("\u2460") || t.startsWith("\u2461") || t.startsWith("\u2462") },
  })), { x: 0.55, y: 2.3, w: 4.1, h: 2.7, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

  // Right: C2
  addCard(s, 5.2, 1.75, 4.4, 3.4, { border: C.teal });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.75, w: 4.4, h: 0.45, fill: { color: C.teal } });
  s.addText("Phase C2: Self-Selection \u81EA\u4E3B\u9009\u62E9", {
    x: 5.3, y: 1.75, w: 4.2, h: 0.45, fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0,
  });

  const c2Lines = [
    "\u6A21\u578B\u81EA\u4E3B\u9009\u62E9\u54EA\u4E9B thinks \u9700\u8981\u538B\u7F29",
    "",
    "\u8BAD\u7EC3\u76EE\u6807\uFF1A",
    "\u2022 Gold = Teacher \u7684 C1 \u7B56\u7565\u8F93\u51FA",
    "\u2022 \u65E0\u663E\u5F0F range \u6307\u5BFC",
    "\u2022 \u6A21\u578B\u5B66\u4F1A\u8BC4\u4F30\u54EA\u4E9B\u4FE1\u606F\u53EF\u4EE5\u5B89\u5168\u538B\u7F29",
    "",
    "\u4ECE C1 \u2192 C2 \u7684\u8FDB\u5316\uFF1A",
    "\u2022 C1: \u7CFB\u7EDF\u544A\u8BC9\u538B\u7F29\u8303\u56F4\uFF0C\u6A21\u578B\u5B66\u6458\u8981",
    "\u2022 C2: \u6A21\u578B\u540C\u65F6\u5B66\u4F1A\u9009\u8303\u56F4+\u5199\u6458\u8981",
  ];
  s.addText(c2Lines.map((t, i) => ({
    text: t, options: { breakLine: i < c2Lines.length - 1, fontSize: 11, color: t.startsWith("\u2022") ? C.sub_text : C.dark_text, bold: t.includes("\u8BAD\u7EC3\u76EE\u6807") || t.includes("\u8FDB\u5316") },
  })), { x: 5.35, y: 2.3, w: 4.1, h: 2.7, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

  // Bottom constraint
  addCard(s, 0.4, 5.25, 9.2, 0.35);
  s.addText([
    { text: "\u7EA6\u675F\uFF1A", options: { bold: true, color: C.accent_red, fontSize: 11 } },
    { text: "\u538B\u7F29\u6BD4 \u2265 2.5:1 | \u6458\u8981\u4EC5\u57FA\u4E8E\u88AB\u538B\u7F29\u7684 thinks\uFF0C\u4E0D\u5F15\u5165\u989D\u5916\u89C6\u89C9\u4FE1\u606F | \u6458\u8981\u4E0D\u80FD\u5305\u542B\u5E27\u4E2D\u53EF\u89C1\u4F46 thinks \u672A\u63D0\u53CA\u7684\u4E8B\u5B9E", options: { color: C.dark_text, fontSize: 10 } },
  ], { x: 0.55, y: 5.25, w: 8.9, h: 0.35, fontFace: "Calibri", valign: "middle", margin: 0 });

  addSlideNumber(s, 6, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 7 — Compression Execution Order
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u538B\u7F29\u6267\u884C\u987A\u5E8F\u4E0E\u7EA6\u675F");

  // Step flow (vertical, left side)
  const steps = [
    { num: "1", title: "Model INPUT", desc: "\u538B\u7F29\u524D\u7684 memory snapshot\n\u4E0D\u5305\u542B\u5F53\u524D think\uFF08\u8FD8\u672A\u751F\u6210\uFF09", color: C.deep },
    { num: "2", title: "Model OUTPUT", desc: "<think>X</think>\n<action>compress</action>\n<summary>Y</summary>", color: C.teal },
    { num: "3", title: "System \u6267\u884C", desc: "a. \u7528 summary Y \u66FF\u6362\u6307\u5B9A\u8303\u56F4\u7684 thinks\nb. \u5C06\u5F53\u524D think X \u8FFD\u52A0\u5230 memory", color: C.light_teal },
  ];

  steps.forEach((st, i) => {
    const sy = 1.1 + i * 1.3;
    addCard(s, 0.5, sy, 5.2, 1.1);
    addIconCircle(s, 0.65, sy + 0.15, 0.45, st.color, st.num);
    s.addText(st.title, { x: 1.25, y: sy + 0.05, w: 4.2, h: 0.3, fontSize: 14, fontFace: "Calibri", color: st.color, bold: true, margin: 0 });
    s.addText(st.desc, { x: 1.25, y: sy + 0.35, w: 4.2, h: 0.65, fontSize: 11, fontFace: "Calibri", color: C.dark_text, margin: 0, lineSpacingMultiple: 1.2 });
    if (i < 2) addArrowDown(s, 3.1, sy + 1.1, 0.2);
  });

  // Right: Hysteresis diagram
  addCard(s, 6.0, 1.1, 3.6, 2.3);
  s.addText("\u8FDF\u6EDE\u673A\u5236 (Hysteresis)", {
    x: 6.15, y: 1.15, w: 3.3, h: 0.35, fontSize: 13, fontFace: "Calibri", color: C.teal, bold: true, margin: 0,
  });

  // Simple bar visualization
  // 80% line
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 1.7, w: 3.0, h: 0.35, fill: { color: C.accent_red, transparency: 70 } });
  s.addText("80% (480 tok) \u2192 \u89E6\u53D1\u538B\u7F29", { x: 6.35, y: 1.7, w: 2.9, h: 0.35, fontSize: 10, fontFace: "Calibri", color: C.accent_red, valign: "middle", margin: 0 });
  // 55% line
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 2.15, w: 2.1, h: 0.35, fill: { color: C.accent_green, transparency: 70 } });
  s.addText("55% (330 tok) \u2192 \u538B\u7F29\u76EE\u6807", { x: 6.35, y: 2.15, w: 2.8, h: 0.35, fontSize: 10, fontFace: "Calibri", color: C.accent_green, valign: "middle", margin: 0 });
  // Arrow
  s.addText("\u2193 \u538B\u7F29\u540E\u964D\u81F3\u6B64\u7EBF\u4EE5\u4E0B", { x: 6.3, y: 2.6, w: 3.0, h: 0.3, fontSize: 9, fontFace: "Calibri", color: C.sub_text, align: "center", margin: 0 });
  s.addText("\u9632\u6B62\u538B\u7F29\u9707\u8361\uFF1A\u907F\u514D\u6BCF\u6B65\u90FD\u89E6\u53D1\u538B\u7F29", { x: 6.3, y: 2.85, w: 3.0, h: 0.3, fontSize: 9, fontFace: "Calibri", color: C.sub_text, italic: true, align: "center", margin: 0 });

  // Right bottom: Key constraint
  addCard(s, 6.0, 3.5, 3.6, 1.9, { border: C.accent_red });
  s.addText("\u5173\u952E\u7EA6\u675F", { x: 6.15, y: 3.55, w: 3.3, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.accent_red, bold: true, margin: 0 });
  const constraints = [
    "\u2022 \u6A21\u578B\u538B\u7F29 INPUT \u4E2D\u7684 recent_thinks",
    "\u2022 \u4E0D\u80FD\u538B\u7F29\u5F53\u524D\u6B63\u5728\u751F\u6210\u7684 think",
    "\u2022 \u6458\u8981\u53EA\u80FD\u5F15\u7528\u88AB\u538B\u7F29 thinks \u4FE1\u606F",
    "\u2022 \u4E0D\u80FD\u5F15\u5165\u5E27\u4E2D\u53EF\u89C1\u4F46 thinks \u672A\u63D0\u53CA\u7684\u4E8B\u5B9E",
    "\u2022 \u7528\u7CBE\u786E tokenizer \u8BA1\u7B97\uFF08\u975E\u4F30\u7B97\uFF09",
  ];
  s.addText(constraints.map((t, i) => ({
    text: t, options: { breakLine: i < constraints.length - 1, fontSize: 10, color: C.dark_text },
  })), { x: 6.15, y: 3.9, w: 3.3, h: 1.4, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.3 });

  addSlideNumber(s, 7, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 8 — Recall Mechanism
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Recall \u673A\u5236\uFF1A\u4E24\u6B65\u68C0\u7D22\u7F16\u6392");

  // Step 1 box
  addCard(s, 0.4, 1.1, 4.2, 2.5, { border: C.deep });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.1, w: 4.2, h: 0.4, fill: { color: C.deep } });
  s.addText("Step 1: Query Generation \u67E5\u8BE2\u751F\u6210", {
    x: 0.5, y: 1.1, w: 4.0, h: 0.4, fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0,
  });
  const s1Lines = [
    { text: "\u8F93\u5165\uFF1A", options: { bold: true, fontSize: 11, color: C.teal, breakLine: true } },
    { text: "\u5F53\u524D\u95EE\u9898 + \u53EF\u89C1\u8BB0\u5FC6\u4E0A\u4E0B\u6587", options: { fontSize: 11, color: C.dark_text, breakLine: true } },
    { text: "(\u4E0D\u63D0\u4F9B gold_answer\uFF01)", options: { fontSize: 10, color: C.accent_red, bold: true, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\u8F93\u51FA\uFF1A", options: { bold: true, fontSize: 11, color: C.teal, breakLine: true } },
    { text: "<think>...</think>", options: { fontSize: 10, color: C.sub_text, breakLine: true } },
    { text: "<action>recall</action>", options: { fontSize: 10, color: C.sub_text, breakLine: true } },
    { text: "<query>{JSON}</query>", options: { fontSize: 10, color: C.sub_text, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\u2192 \u7CFB\u7EDF\u89E3\u6790 query\uFF0C\u641C\u7D22 archive", options: { fontSize: 10, color: C.sub_text, italic: true } },
  ];
  s.addText(s1Lines, { x: 0.55, y: 1.6, w: 3.9, h: 1.9, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

  // Arrow
  addArrowRight(s, 4.6, 2.35, 0.8);

  // Step 2 box
  addCard(s, 5.4, 1.1, 4.2, 2.5, { border: C.teal });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: 1.1, w: 4.2, h: 0.4, fill: { color: C.teal } });
  s.addText("Step 2: Post-Recall Response \u68C0\u7D22\u540E\u56DE\u7B54", {
    x: 5.5, y: 1.1, w: 4.0, h: 0.4, fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0,
  });
  const s2Lines = [
    { text: "\u8F93\u5165\uFF1A", options: { bold: true, fontSize: 11, color: C.teal, breakLine: true } },
    { text: "\u8BB0\u5FC6 + recall_result (top-4)", options: { fontSize: 11, color: C.dark_text, breakLine: true } },
    { text: "+ recalled_frames (4s\u89C6\u9891\u5E27)", options: { fontSize: 11, color: C.dark_text, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\u8F93\u51FA\uFF1A", options: { bold: true, fontSize: 11, color: C.teal, breakLine: true } },
    { text: "<action>response</action>", options: { fontSize: 10, color: C.sub_text, breakLine: true } },
    { text: "<response>\u56DE\u7B54\u5185\u5BB9</response>", options: { fontSize: 10, color: C.sub_text, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\u65E0 <think> \u6807\u7B7E \u2192 \u907F\u514D double write", options: { fontSize: 10, color: C.accent_orange, bold: true } },
  ];
  s.addText(s2Lines, { x: 5.55, y: 1.6, w: 3.9, h: 1.9, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

  // Anti-leakage callout
  addCard(s, 0.4, 3.8, 4.8, 0.5, { border: C.accent_red });
  s.addText([
    { text: "\u9632\u6CC4\u6F0F\uFF1A", options: { bold: true, color: C.accent_red, fontSize: 11 } },
    { text: "Query \u7EDD\u5BF9\u4E0D\u80FD\u5305\u542B\u7B54\u6848\u5173\u952E\u8BCD\uFF0C\u4EC5\u5305\u542B\u68C0\u7D22\u610F\u56FE", options: { color: C.dark_text, fontSize: 11 } },
  ], { x: 0.55, y: 3.8, w: 4.5, h: 0.5, fontFace: "Calibri", valign: "middle", margin: 0 });

  // Retrieval noise distribution
  addCard(s, 0.4, 4.5, 9.2, 1.0);
  s.addText("\u68C0\u7D22\u7ED3\u679C\u566A\u58F0\u5206\u5E03 (\u8BAD\u7EC3\u65F6\u6A21\u62DF\u4E0D\u5B8C\u7F8E\u68C0\u7D22)", {
    x: 0.55, y: 4.5, w: 4, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.teal, bold: true, margin: 0,
  });

  // Noise bars
  const noises = [
    { label: "70% Oracle", desc: "\u6B63\u786E\u8BC1\u636E\u5728 rank 1", w: 3.5, color: C.accent_green },
    { label: "20% Noisy", desc: "\u6B63\u786E\u5728 rank 2-4", w: 1.0, color: C.accent_orange },
    { label: "5% All-Wrong", desc: "\u4EC5\u5E72\u6270\u9879", w: 0.25, color: C.accent_red },
    { label: "5% Failure", desc: "\u7A7A\u7ED3\u679C", w: 0.25, color: C.sub_text },
  ];
  let nx = 0.55;
  noises.forEach((n) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx, y: 4.85, w: n.w, h: 0.2, fill: { color: n.color } });
    nx += n.w + 0.02;
  });
  // Labels
  s.addText(noises.map((n, i) => ({
    text: `${n.label}: ${n.desc}${i < noises.length - 1 ? "  |  " : ""}`,
    options: { fontSize: 9, color: C.sub_text },
  })), { x: 0.55, y: 5.1, w: 8.5, h: 0.25, fontFace: "Calibri", margin: 0 });

  // Source table (right)
  s.addText("\u68C0\u7D22\u6765\u6E90 (\u4EC5\u5B66\u751F\u53EF\u89C1)", { x: 5.8, y: 3.85, w: 3.5, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.teal, bold: true, margin: 0 });
  s.addTable([
    [{ text: "\u6765\u6E90", options: { fill: { color: C.teal }, color: C.white, bold: true, fontSize: 9 } },
     { text: "\u5185\u5BB9", options: { fill: { color: C.teal }, color: C.white, bold: true, fontSize: 9 } }],
    [{ text: "student_think", options: { fontSize: 9 } }, { text: "\u5B66\u751F\u81EA\u5DF1\u7684\u89C2\u5BDF\u6587\u672C", options: { fontSize: 9 } }],
    [{ text: "compressed_summary", options: { fontSize: 9 } }, { text: "\u5B66\u751F\u751F\u6210\u7684\u538B\u7F29\u6458\u8981", options: { fontSize: 9 } }],
    [{ text: "historical_frames", options: { fontSize: 9 } }, { text: "4s \u539F\u59CB\u89C6\u9891\u5E27", options: { fontSize: 9 } }],
  ], { x: 5.8, y: 4.15, w: 3.7, h: 0.7, border: { pt: 0.5, color: C.light_gray }, colW: [1.5, 2.2], fontSize: 9, fontFace: "Calibri" });

  addSlideNumber(s, 8, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 9 — Position Encoding
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u6D41\u5F0F\u89C6\u9891\u4F4D\u7F6E\u7F16\u7801\uFF1AZone-Based + Temporal MROPE");

  // Top: Zone Layout
  s.addText("Zone-Based \u8F93\u5165\u5E03\u5C40 (\u89C6\u9891\u524D\u7F6E)", {
    x: 0.6, y: 1.0, w: 4, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.deep, bold: true, margin: 0,
  });

  const zones = [
    { label: "Zone A: System Prompt", tok: "~150 tok", note: "fixed, KV reuse", color: C.sub_text, w: 1.2 },
    { label: "Zone B: Visual Window", tok: "~1,536 tok", note: "stable position", color: C.deep, w: 3.5 },
    { label: "Zone C: Memory Text", tok: "~600-1,350 tok", note: "appends only", color: C.teal, w: 2.8 },
    { label: "Zone D: User Input", tok: "~50 tok", note: "always new", color: C.light_teal, w: 0.8 },
  ];

  let zx = 0.6;
  zones.forEach((z) => {
    s.addShape(pres.shapes.RECTANGLE, { x: zx, y: 1.4, w: z.w, h: 0.7, fill: { color: z.color, transparency: 25 }, line: { color: z.color, width: 1.2 } });
    s.addText(z.label, { x: zx + 0.05, y: 1.4, w: z.w - 0.1, h: 0.25, fontSize: 8, fontFace: "Calibri", color: z.color, bold: true, margin: 0 });
    s.addText(z.tok, { x: zx + 0.05, y: 1.62, w: z.w - 0.1, h: 0.2, fontSize: 8, fontFace: "Calibri", color: C.dark_text, margin: 0 });
    s.addText(z.note, { x: zx + 0.05, y: 1.82, w: z.w - 0.1, h: 0.2, fontSize: 7, fontFace: "Calibri", color: C.sub_text, italic: true, margin: 0 });
    zx += z.w + 0.08;
  });

  // Key change annotation
  addCard(s, 0.6, 2.3, 8.8, 0.5);
  s.addText([
    { text: "\u5173\u952E\u53D8\u66F4\uFF1A", options: { bold: true, color: C.accent_orange, fontSize: 11 } },
    { text: "\u89C6\u9891\u79FB\u81F3\u524D\u7AEF (Zone B)\uFF0C\u4F4D\u7F6E\u7D22\u5F15\u7A33\u5B9A \u2192 Zone A+B \u7684 KV cache \u53EF\u8DE8\u6B65\u590D\u7528 \u2192 \u6BCF\u6B65\u4EC5\u9700\u91CD\u7B97 Zone C+D \u2192 ", options: { color: C.dark_text, fontSize: 10 } },
    { text: "90% prefill \u51CF\u5C11", options: { bold: true, color: C.teal, fontSize: 11 } },
  ], { x: 0.75, y: 2.3, w: 8.5, h: 0.5, fontFace: "Calibri", valign: "middle", margin: 0 });

  // Bottom: Temporal MROPE
  s.addText("Temporal-Aligned MROPE \u65F6\u95F4\u5BF9\u9F50\u4F4D\u7F6E\u7F16\u7801", {
    x: 0.6, y: 2.95, w: 6, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.teal, bold: true, margin: 0,
  });

  // MROPE diagram - video token
  addCard(s, 0.5, 3.35, 4.2, 1.8);
  s.addText("Qwen2.5-VL 3D RoPE: (temporal, height, width)", {
    x: 0.65, y: 3.4, w: 3.9, h: 0.25, fontSize: 10, fontFace: "Calibri", color: C.sub_text, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 3.72, w: 3.8, h: 0.55, fill: { color: C.deep, transparency: 85 }, line: { color: C.deep, width: 1 } });
  s.addText([
    { text: "Video frame (t=10s)\n", options: { bold: true, fontSize: 10, color: C.deep, breakLine: true } },
    { text: "temporal=encode(10), h=grid[0], w=grid[1]", options: { fontSize: 9, color: C.dark_text } },
  ], { x: 0.8, y: 3.72, w: 3.6, h: 0.55, fontFace: "Calibri", valign: "middle", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.37, w: 3.8, h: 0.55, fill: { color: C.teal, transparency: 85 }, line: { color: C.teal, width: 1 } });
  s.addText([
    { text: "Think text (t=10s)\n", options: { bold: true, fontSize: 10, color: C.teal, breakLine: true } },
    { text: "temporal=encode(10), h=seq_idx, w=seq_idx", options: { fontSize: 9, color: C.dark_text } },
  ], { x: 0.8, y: 4.37, w: 3.6, h: 0.55, fontFace: "Calibri", valign: "middle", margin: 0 });

  // = sign between
  s.addText("\u2191 SAME temporal \u2191", {
    x: 1.5, y: 4.22, w: 2.0, h: 0.2, fontSize: 9, fontFace: "Calibri", color: C.accent_orange, bold: true, align: "center", margin: 0,
  });

  // Right: explanation
  addCard(s, 5.0, 3.35, 4.6, 1.8, { border: C.teal });
  s.addText("\u8DE8\u6A21\u6001\u6CE8\u610F\u529B\u589E\u5F3A", {
    x: 5.15, y: 3.4, w: 4.3, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.teal, bold: true, margin: 0,
  });
  const mropeExplain = [
    "\u89C6\u9891\u5E27\u548C\u5BF9\u5E94\u65F6\u523B\u7684 think \u6587\u672C\u5171\u4EAB\u76F8\u540C\u7684\u65F6\u95F4\u7EF4\u5EA6\u4F4D\u7F6E\u7F16\u7801",
    "",
    "\u5373\u4F7F\u5728\u5E8F\u5217\u4E2D\u76F8\u8DDD 1000+ tokens\uFF0C",
    "\u65F6\u95F4\u7EF4\u5EA6\u8DDD\u79BB = 0",
    "",
    "\u6548\u679C\uFF1A",
    "\u2022 \u89C6\u89C9-\u6587\u672C\u8DE8\u6A21\u6001 attention \u663E\u8457\u589E\u5F3A",
    "\u2022 \u6A21\u578B\u66F4\u5BB9\u6613\u5C06 think \u4E0E\u5BF9\u5E94\u5E27\u5173\u8054",
    "\u2022 \u538B\u7F29\u6458\u8981\u4E0E\u539F\u59CB\u65F6\u95F4\u8303\u56F4\u5BF9\u9F50",
  ];
  s.addText(mropeExplain.map((t, i) => ({
    text: t, options: { breakLine: i < mropeExplain.length - 1, fontSize: 10, color: t.startsWith("\u2022") ? C.sub_text : C.dark_text, bold: t.includes("\u6548\u679C") },
  })), { x: 5.15, y: 3.75, w: 4.3, h: 1.3, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.15 });

  addSlideNumber(s, 9, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 10 — KV Cache Partitioned
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "KV Cache \u5206\u533A\u7B56\u7565\uFF1A\u6309 Zone \u589E\u91CF\u66F4\u65B0");

  // Three scenario columns
  const scenarios = [
    { title: "Normal Step\n\u666E\u901A\u6B65", zones: ["0 tok (reuse)", "~128 tok", "~50 tok", "~50 tok"], total: "~228 tok", color: C.accent_green },
    { title: "Compress Step\n\u538B\u7F29\u6B65", zones: ["0 tok (reuse)", "~128 tok", "~500 tok \u2605", "~50 tok"], total: "~678 tok", color: C.accent_orange },
    { title: "Recall Step\n\u68C0\u7D22\u6B65", zones: ["0 tok (reuse)", "~384 tok", "~200 tok", "~50 tok"], total: "~634 tok", color: C.accent_purple },
  ];

  scenarios.forEach((sc, i) => {
    const sx = 0.4 + i * 3.2;
    addCard(s, sx, 1.05, 2.8, 3.3);
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 1.05, w: 2.8, h: 0.55, fill: { color: sc.color } });
    s.addText(sc.title, { x: sx, y: 1.05, w: 2.8, h: 0.55, fontSize: 12, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    const zoneLabels = ["Zone A (System)", "Zone B (Visual)", "Zone C (Memory)", "Zone D (User)"];
    const zoneColors = [C.sub_text, C.deep, C.teal, C.light_teal];
    sc.zones.forEach((z, j) => {
      const zy = 1.7 + j * 0.45;
      s.addShape(pres.shapes.RECTANGLE, { x: sx + 0.1, y: zy, w: 0.08, h: 0.35, fill: { color: zoneColors[j] } });
      s.addText(zoneLabels[j], { x: sx + 0.25, y: zy, w: 1.3, h: 0.18, fontSize: 8, fontFace: "Calibri", color: zoneColors[j], margin: 0 });
      s.addText(z, { x: sx + 0.25, y: zy + 0.16, w: 2.3, h: 0.18, fontSize: 9, fontFace: "Calibri", color: C.dark_text, bold: z.includes("\u2605"), margin: 0 });
    });

    // Total
    s.addShape(pres.shapes.RECTANGLE, { x: sx + 0.1, y: 3.55, w: 2.6, h: 0.03, fill: { color: sc.color } });
    s.addText([
      { text: "Prefill: ", options: { fontSize: 11, color: C.sub_text } },
      { text: sc.total, options: { fontSize: 14, color: sc.color, bold: true } },
    ], { x: sx + 0.1, y: 3.6, w: 2.6, h: 0.35, fontFace: "Calibri", align: "center", valign: "middle", margin: 0 });
  });

  // Comparison bar
  addCard(s, 0.4, 4.55, 9.2, 0.85, { border: C.teal });

  s.addText("\u5BF9\u6BD4\uFF1A\u4F20\u7EDF\u65B9\u6CD5\u6BCF\u6B65 full prefill", {
    x: 0.55, y: 4.6, w: 4, h: 0.25, fontSize: 11, fontFace: "Calibri", color: C.sub_text, margin: 0,
  });
  // Old bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.9, w: 6.0, h: 0.2, fill: { color: C.accent_red, transparency: 30 } });
  s.addText("~3,086 tok (full prefill)", { x: 0.6, y: 4.9, w: 5.8, h: 0.2, fontSize: 9, fontFace: "Calibri", color: C.accent_red, margin: 0, valign: "middle" });
  // New bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.12, w: 0.6, h: 0.2, fill: { color: C.accent_green, transparency: 30 } });
  s.addText("~228 tok", { x: 0.6, y: 5.12, w: 2.0, h: 0.2, fontSize: 9, fontFace: "Calibri", color: C.accent_green, margin: 0, valign: "middle" });

  // Stats
  s.addText([
    { text: "90% reduction  |  ", options: { fontSize: 16, color: C.teal, bold: true } },
    { text: "H100 ~200ms/step  |  2s budget  |  \u5B9E\u65F6\u63A8\u7406\u2714", options: { fontSize: 12, color: C.dark_text } },
  ], { x: 6.7, y: 4.6, w: 3.0, h: 0.8, fontFace: "Calibri", valign: "middle", align: "center", margin: 0 });

  addSlideNumber(s, 10, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 11 — Data Pipeline Overview
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u4E94\u9636\u6BB5\u6570\u636E\u6784\u9020\u6D41\u6C34\u7EBF (Agent Data v5)");

  const passes = [
    { num: "Pass 1", title: "Teacher Evidence Graph", time: "397B, ~4h", desc: "24\u5E27\u6ED1\u52A8\u7A97\u53E3 + \u5386\u53F2caption\n\u2192 \u9010chunk\u539F\u5B50\u4E8B\u5B9E (\u5B9E\u4F53/OCR/\u6570\u5B57)", color: C.deep },
    { num: "Pass 2", title: "Question-Blind Rollout", time: "397B, ~4h", desc: "Memory + Visual (\u65E0\u95EE\u9898!)\n\u2192 Student thinks + \u538B\u7F29\u4E8B\u4EF6 + \u5FEB\u7167", color: C.teal },
    { num: "Pass 3", title: "Task Planning", time: "397B, ~1h", desc: "Teacher\u8BC1\u636E + Student rollout\n\u2192 \u4EFB\u52A1\u5B9A\u4E49 + gold action + \u7B54\u6848", color: C.light_teal },
    { num: "Pass 4", title: "Question-Aware Forks", time: "397B, ~2h", desc: "\u7F13\u5B58\u5FEB\u7167 + \u4EFB\u52A1\n\u2192 Per-timestep SFT \u6837\u672C", color: C.deep },
    { num: "Pass 5", title: "Verify + Filter", time: "Rules, ~30min", desc: "5\u7C7B\u9A8C\u8BC1 \u2192 \u6700\u7EC8 .jsonl\n\u4FE1\u606F\u6D41/\u6700\u5C0F\u6027/\u63A5\u5730/\u683C\u5F0F/\u96BE\u5EA6", color: C.teal },
  ];

  passes.forEach((p, i) => {
    const py = 1.0 + i * 0.82;
    addCard(s, 0.4, py, 6.6, 0.72);
    // Number circle
    addIconCircle(s, 0.55, py + 0.1, 0.42, p.color, p.num.split(" ")[1]);
    // Title
    s.addText(`${p.num}: ${p.title}`, { x: 1.1, y: py + 0.02, w: 3.5, h: 0.25, fontSize: 12, fontFace: "Calibri", color: p.color, bold: true, margin: 0 });
    s.addText(p.desc, { x: 1.1, y: py + 0.27, w: 4.0, h: 0.42, fontSize: 10, fontFace: "Calibri", color: C.dark_text, margin: 0, lineSpacingMultiple: 1.1 });
    // Time badge
    s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: py + 0.15, w: 1.4, h: 0.35, fill: { color: p.color, transparency: 80 }, line: { color: p.color, width: 0.5 } });
    s.addText(p.time, { x: 5.4, y: py + 0.15, w: 1.4, h: 0.35, fontSize: 9, fontFace: "Calibri", color: p.color, align: "center", valign: "middle", margin: 0 });

    if (i < 4) addArrowDown(s, 3.5, py + 0.72, 0.1);
  });

  // Right: Key principle
  addCard(s, 7.2, 1.0, 2.5, 4.1, { border: C.accent_orange });
  s.addText("\u6838\u5FC3\u539F\u5219", { x: 7.3, y: 1.05, w: 2.3, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.accent_orange, bold: true, margin: 0 });

  const principles = [
    "Question-Blind\n\u8BBE\u8BA1",
    "Pass 2 \u5B8C\u5168\u4E0D\u770B\u4EFB\u4F55\u95EE\u9898\uFF0C\u4FDD\u8BC1\u8BB0\u5FC6\u4E0D\u88AB\u672A\u6765\u95EE\u9898\u6C61\u67D3",
    "",
    "Teacher-Student\n\u5206\u79BB",
    "Teacher \u8BC1\u636E\u4EC5\u7528\u4E8E\u8BC4\u4F30\uFF0C\u5B66\u751F\u6C38\u8FDC\u770B\u4E0D\u5230 teacher caption",
    "",
    "\u5FEB\u7167\u590D\u7528",
    "Pass 2 \u7684 memory \u5FEB\u7167\u88AB Pass 4 \u590D\u7528\uFF0C\u907F\u514D\u91CD\u590D\u6EDA\u52A8",
  ];
  s.addText(principles.map((t, i) => ({
    text: t, options: {
      breakLine: i < principles.length - 1,
      fontSize: t === "" ? 4 : (t.includes("\u8BBE\u8BA1") || t.includes("\u5206\u79BB") || t.includes("\u590D\u7528")) ? 11 : 9,
      color: (t.includes("\u8BBE\u8BA1") || t.includes("\u5206\u79BB") || t.includes("\u590D\u7528")) ? C.accent_orange : C.dark_text,
      bold: t.includes("\u8BBE\u8BA1") || t.includes("\u5206\u79BB") || t.includes("\u590D\u7528"),
    },
  })), { x: 7.3, y: 1.45, w: 2.3, h: 3.5, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

  addSlideNumber(s, 11, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 12 — Pass 1-2 Detail
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Pass 1-2: \u6559\u5E08\u8BC1\u636E\u56FE & \u65E0\u95EE\u9898\u6EDA\u52A8");

  // Pass 1 row
  addCard(s, 0.4, 1.05, 9.2, 1.8);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.05, w: 0.08, h: 1.8, fill: { color: C.deep } });
  addIconCircle(s, 0.65, 1.15, 0.4, C.deep, "1");
  s.addText("Pass 1: Teacher Evidence Graph \u6559\u5E08\u8BC1\u636E\u56FE", {
    x: 1.15, y: 1.1, w: 5, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.deep, bold: true, margin: 0,
  });

  // Flow diagram for Pass 1
  const p1Boxes = [
    { x: 0.65, label: "\u89C6\u9891\u5206\u5757\n(2s chunks)" },
    { x: 2.5, label: "24\u5E27\u6ED1\u52A8\u7A97\u53E3\n+ \u5386\u53F2 caption" },
    { x: 4.5, label: "397B Teacher\n\u6559\u5E08\u6A21\u578B" },
    { x: 6.6, label: "\u539F\u5B50\u4E8B\u5B9E\n(\u5B9E\u4F53/OCR/\u6570\u5B57)" },
  ];
  p1Boxes.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: b.x, y: 1.55, w: 1.6, h: 0.65, fill: { color: C.deep, transparency: 85 }, line: { color: C.deep, width: 1 } });
    s.addText(b.label, { x: b.x, y: 1.55, w: 1.6, h: 0.65, fontSize: 9, fontFace: "Calibri", color: C.dark_text, align: "center", valign: "middle", margin: 0 });
    if (i < 3) addArrowRight(s, b.x + 1.6, 1.88, 0.7);
  });

  s.addText("\u4EC5\u7528\u4E8E\u540E\u7EED\u8BC4\u4F30\uFF0C\u5B66\u751F\u6C38\u8FDC\u770B\u4E0D\u5230\u6559\u5E08\u8BC1\u636E", {
    x: 6.6, y: 2.3, w: 3.0, h: 0.3, fontSize: 9, fontFace: "Calibri", color: C.accent_red, italic: true, margin: 0,
  });

  // Pass 2 row
  addCard(s, 0.4, 3.05, 9.2, 2.4);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 3.05, w: 0.08, h: 2.4, fill: { color: C.teal } });
  addIconCircle(s, 0.65, 3.15, 0.4, C.teal, "2");
  s.addText("Pass 2: Question-Blind Rollout \u65E0\u95EE\u9898\u6EDA\u52A8", {
    x: 1.15, y: 3.1, w: 5, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.teal, bold: true, margin: 0,
  });

  // Circular flow for Pass 2
  const p2Boxes = [
    { x: 0.65, y: 3.6, label: "Memory State\n\u5F53\u524D\u8BB0\u5FC6\u72B6\u6001" },
    { x: 2.5, y: 3.6, label: "Visual Window\n24\u5E27 + Memory" },
    { x: 4.4, y: 3.6, label: "Generate Think\n\u751F\u6210\u601D\u8003\u8BB0\u5FC6" },
    { x: 6.3, y: 3.6, label: "Update Memory\n\u8FFD\u52A0 think" },
  ];
  p2Boxes.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: b.x, y: b.y, w: 1.6, h: 0.6, fill: { color: C.teal, transparency: 85 }, line: { color: C.teal, width: 1 } });
    s.addText(b.label, { x: b.x, y: b.y, w: 1.6, h: 0.6, fontSize: 9, fontFace: "Calibri", color: C.dark_text, align: "center", valign: "middle", margin: 0 });
    if (i < 3) addArrowRight(s, b.x + 1.6, b.y + 0.3, 0.7);
  });

  // Compression branch
  s.addShape(pres.shapes.RECTANGLE, { x: 8.2, y: 3.6, w: 1.2, h: 0.6, fill: { color: C.accent_orange, transparency: 80 }, line: { color: C.accent_orange, width: 1 } });
  s.addText("\u89E6\u53D1\u538B\u7F29?\n\u226580%", { x: 8.2, y: 3.6, w: 1.2, h: 0.6, fontSize: 9, fontFace: "Calibri", color: C.accent_orange, align: "center", valign: "middle", bold: true, margin: 0 });
  addArrowRight(s, 7.9, 3.9, 0.3);

  // Description
  const p2Desc = [
    "\u6838\u5FC3\u6D41\u7A0B\uFF1A\u6BCF\u4E2A chunk \u751F\u6210\u4E00\u6B21 think \u2192 \u66F4\u65B0 memory \u2192 \u68C0\u67E5\u538B\u7F29\u89E6\u53D1",
    "",
    "\u2022 \u5B8C\u5168\u4E0D\u770B\u4EFB\u4F55\u95EE\u9898 (Question-Blind)\uFF0C\u786E\u4FDD\u8BB0\u5FC6\u7684\u7EAF\u51C0\u6027",
    "\u2022 \u6BCF\u4E2A chunk \u4FDD\u5B58\u4E00\u4E2A memory snapshot\uFF0C\u4F9B Pass 4 \u590D\u7528",
    "\u2022 \u538B\u7F29\u4E8B\u4EF6\u81EA\u7136\u53D1\u751F\uFF1A\u5F53 tokens \u2265 80% budget \u65F6\u89E6\u53D1",
    "\u2022 \u591A\u7EF4\u5EA6\u8BC4\u5206\u9009\u62E9\u6700\u4F18\u538B\u7F29\u8303\u56F4\uFF0C\u751F\u6210\u6458\u8981\u66FF\u6362\u65E7 thinks",
  ];
  s.addText(p2Desc.map((t, i) => ({
    text: t, options: { breakLine: i < p2Desc.length - 1, fontSize: t === "" ? 4 : t.startsWith("\u2022") ? 10 : 11, color: t.startsWith("\u2022") ? C.sub_text : C.dark_text, bold: !t.startsWith("\u2022") && t !== "" },
  })), { x: 0.65, y: 4.35, w: 8.8, h: 1.0, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.15 });

  addSlideNumber(s, 12, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 13 — Pass 3-4 Detail
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Pass 3-4: \u4EFB\u52A1\u89C4\u5212 & \u6837\u672C\u5206\u53C9");

  // Pass 3: Action Minimality decision tree
  addCard(s, 0.4, 1.0, 9.2, 2.3);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.0, w: 0.08, h: 2.3, fill: { color: C.light_teal } });
  addIconCircle(s, 0.65, 1.1, 0.4, C.light_teal, "3");
  s.addText("Pass 3: Action Minimality \u52A8\u4F5C\u6700\u5C0F\u6027\u51B3\u7B56\u6811", {
    x: 1.15, y: 1.05, w: 6, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.light_teal, bold: true, margin: 0,
  });

  // Decision tree using boxes and arrows
  const dt = [
    { x: 0.6, y: 1.55, w: 2.0, label: "\u7B54\u6848\u5728\u5F53\u524D\u89C6\u89C9\u7A97\u53E3?", color: C.deep },
    { x: 3.0, y: 1.55, w: 1.2, label: "YES\n\u2192 response", color: C.accent_green },
    { x: 0.6, y: 2.15, w: 2.0, label: "\u7B54\u6848\u5728 recent_thinks?", color: C.deep },
    { x: 3.0, y: 2.15, w: 1.2, label: "YES\n\u2192 response", color: C.accent_green },
    { x: 4.6, y: 1.55, w: 2.0, label: "\u7B54\u6848\u5728\u538B\u7F29\u6458\u8981\u4E2D?", color: C.teal },
    { x: 7.0, y: 1.55, w: 1.2, label: "YES\n\u2192 response", color: C.accent_green },
    { x: 4.6, y: 2.15, w: 2.0, label: "\u7B54\u6848\u5728 archive \u4E2D?", color: C.teal },
    { x: 7.0, y: 2.15, w: 1.2, label: "YES\n\u2192 recall", color: C.accent_orange },
    { x: 8.5, y: 2.15, w: 1.0, label: "NO\n\u2192 uncertain", color: C.accent_red },
  ];

  dt.forEach(d => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: d.x, y: d.y, w: d.w, h: 0.5,
      fill: { color: d.color, transparency: d.w <= 1.2 ? 70 : 85 },
      line: { color: d.color, width: 1 },
    });
    s.addText(d.label, {
      x: d.x, y: d.y, w: d.w, h: 0.5,
      fontSize: 8, fontFace: "Calibri", color: C.dark_text, align: "center", valign: "middle", margin: 0, bold: d.w <= 1.2,
    });
  });

  // Connecting arrows (simplified)
  addArrowRight(s, 2.6, 1.8, 0.35);
  addArrowRight(s, 2.6, 2.4, 0.35);
  addArrowRight(s, 6.6, 1.8, 0.35);
  addArrowRight(s, 6.6, 2.4, 0.35);
  addArrowRight(s, 8.2, 2.4, 0.25);

  // "NO" down arrows
  s.addText("NO \u2193", { x: 1.2, y: 2.05, w: 0.5, h: 0.12, fontSize: 7, fontFace: "Calibri", color: C.accent_red, align: "center", margin: 0 });
  s.addText("NO \u2192", { x: 2.6, y: 2.25, w: 0.4, h: 0.12, fontSize: 7, fontFace: "Calibri", color: C.accent_red, margin: 0 });

  // Key principle
  s.addText([
    { text: "\u6838\u5FC3\uFF1A", options: { bold: true, color: C.accent_orange, fontSize: 10 } },
    { text: "\u5982\u679C\u538B\u7F29\u6458\u8981\u5305\u542B\u7B54\u6848 \u2192 \u76F4\u63A5 response\uFF0C\u4E0D\u9700\u8981 recall\uFF01\u8FD9\u4FDD\u8BC1\u4E86\u538B\u7F29\u6458\u8981\u7684\u5B9E\u7528\u6027", options: { color: C.dark_text, fontSize: 10 } },
  ], { x: 0.6, y: 2.75, w: 8.8, h: 0.3, fontFace: "Calibri", margin: 0 });

  // Pass 4 section
  addCard(s, 0.4, 3.3, 9.2, 2.2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 3.3, w: 0.08, h: 2.2, fill: { color: C.deep } });
  addIconCircle(s, 0.65, 3.4, 0.4, C.deep, "4");
  s.addText("Pass 4: Question-Aware Forks \u6837\u672C\u5206\u53C9\u751F\u6210", {
    x: 1.15, y: 3.35, w: 6, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.deep, bold: true, margin: 0,
  });

  // Fork flow
  const forkBoxes = [
    { x: 0.65, label: "Cached\nSnapshot\n\u7F13\u5B58\u5FEB\u7167", color: C.teal },
    { x: 2.3, label: "\u6CE8\u5165\u95EE\u9898\n\u5230\u5FEB\u7167\u4E2D", color: C.accent_orange },
    { x: 4.0, label: "\u751F\u6210\u8F93\u51FA\nthink+action", color: C.deep },
  ];
  forkBoxes.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: b.x, y: 3.85, w: 1.4, h: 0.7, fill: { color: b.color, transparency: 82 }, line: { color: b.color, width: 1 } });
    s.addText(b.label, { x: b.x, y: 3.85, w: 1.4, h: 0.7, fontSize: 9, fontFace: "Calibri", color: C.dark_text, align: "center", valign: "middle", margin: 0 });
    if (i < 2) addArrowRight(s, b.x + 1.4, 4.2, 0.7);
  });

  // Output types
  s.addText("\u2192", { x: 5.35, y: 4.05, w: 0.3, h: 0.3, fontSize: 16, color: C.teal, align: "center", margin: 0 });

  const types = [
    { label: "silent", color: C.sub_text },
    { label: "response", color: C.accent_green },
    { label: "recall", color: C.accent_orange },
    { label: "compress", color: C.deep },
  ];
  types.forEach((t, i) => {
    const tx = 5.7 + i * 1.05;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 3.95, w: 0.95, h: 0.4, fill: { color: t.color, transparency: 70 }, line: { color: t.color, width: 1 } });
    s.addText(t.label, { x: tx, y: 3.95, w: 0.95, h: 0.4, fontSize: 9, fontFace: "Calibri", color: t.color, align: "center", valign: "middle", bold: true, margin: 0 });
  });

  // Additional details
  const p4Details = [
    "\u2022 Pending questions: \u6BCF\u4E2A\u5F85\u7B54\u95EE\u9898\u751F\u6210 3 \u4E2A\u8BAD\u7EC3\u6837\u672C (setup \u2192 wait \u2192 trigger)",
    "\u2022 \u52A8\u4F5C\u4F18\u5148\u7EA7: user response > pending trigger > compress > silent",
    "\u2022 Recall \u751F\u6210\u4E24\u4E2A\u6837\u672C: recall_query + recall_response\uFF0Cpost-recall \u65E0 <think>",
    "\u2022 \u6BCF\u4E2A\u6837\u672C\u7EDF\u4E00\u683C\u5F0F: input{system, memory, visual_window, user_input} + output{think, action, payload}",
  ];
  s.addText(p4Details.map((t, i) => ({
    text: t, options: { breakLine: i < p4Details.length - 1, fontSize: 10, color: C.dark_text },
  })), { x: 0.65, y: 4.55, w: 8.8, h: 0.85, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.2 });

  addSlideNumber(s, 13, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 14 — Pass 5 Verification
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "Pass 5: \u4E94\u7C7B\u8D28\u91CF\u9A8C\u8BC1");

  const verCards = [
    {
      title: "\u2460 \u4FE1\u606F\u6D41\u5408\u6CD5\u6027",
      items: ["\u5F53\u524D\u52A8\u4F5C\u4EC5\u4F9D\u8D56 ask_time \u53EF\u89C1\u4FE1\u606F", "Recall query \u4E0D\u542B\u7B54\u6848", "\u63D0\u95EE\u524D\u8BB0\u5FC6\u662F question-blind", "Recall result \u4EC5\u6765\u81EA\u5B66\u751F archive"],
      color: C.deep, x: 0.4, y: 1.05, w: 4.35,
    },
    {
      title: "\u2461 \u52A8\u4F5C\u6700\u5C0F\u6027",
      items: ["\u7B54\u6848\u53EF\u89C1 \u2192 \u5FC5\u987B response", "\u7B54\u6848\u5728\u6458\u8981 \u2192 response \u975E recall", "\u65E0\u6CD5\u56DE\u7B54 \u2192 \u6807\u8BB0 uncertain"],
      color: C.teal, x: 5.25, y: 1.05, w: 4.35,
    },
    {
      title: "\u2462 \u4E8B\u5B9E\u63A5\u5730",
      items: ["Think \u7531\u5F53\u524D\u5E27\u652F\u6301", "\u6458\u8981\u4EC5\u542B\u88AB\u538B\u7F29 thinks \u4FE1\u606F", "\u7981\u6B62\u58F0\u97F3/\u6C14\u5473/\u60C5\u611F/\u610F\u56FE\u63A8\u65AD", "Response \u6709\u660E\u786E\u8BC1\u636E\u652F\u6491"],
      color: C.light_teal, x: 0.4, y: 2.85, w: 2.85,
    },
    {
      title: "\u2463 \u683C\u5F0F\u4E0E\u957F\u5EA6",
      items: ["Think: 40-60 tokens", "Summary \u538B\u7F29\u6BD4 \u2265 2.5:1", "Query JSON \u6709\u6548\u4E14\u65E0 leakage", "\u7279\u6B8A\u6807\u7B7E\u683C\u5F0F\u4E25\u683C\u5339\u914D"],
      color: C.accent_orange, x: 3.58, y: 2.85, w: 2.85,
    },
    {
      title: "\u2464 \u96BE\u5EA6\u6807\u6CE8",
      items: ["current_visible (Easy)", "memory_response (Medium)", "recall_required (Hard)", "unanswerable (Very Hard)"],
      color: C.accent_purple, x: 6.75, y: 2.85, w: 2.85,
    },
  ];

  verCards.forEach(c => {
    addCard(s, c.x, c.y, c.w, c.y < 2 ? 1.55 : 1.55, { border: c.color });
    s.addText(c.title, { x: c.x + 0.12, y: c.y + 0.05, w: c.w - 0.24, h: 0.3, fontSize: 12, fontFace: "Calibri", color: c.color, bold: true, margin: 0 });
    s.addText(c.items.map((t, i) => ({
      text: t, options: { bullet: true, breakLine: i < c.items.length - 1, fontSize: 10, color: C.dark_text },
    })), { x: c.x + 0.12, y: c.y + 0.38, w: c.w - 0.24, h: 1.1, fontFace: "Calibri", valign: "top", margin: 0, lineSpacingMultiple: 1.15 });
  });

  // Bottom stats
  addCard(s, 0.4, 4.6, 9.2, 0.85, { border: C.accent_green });
  s.addText("\u9A8C\u6536\u6807\u51C6", {
    x: 0.55, y: 4.65, w: 2, h: 0.3, fontSize: 13, fontFace: "Calibri", color: C.accent_green, bold: true, margin: 0,
  });
  const acceptance = [
    "p99 < 14,746 tokens (16,384 \u00D7 90%)",
    "overflow < 0.5% (\u8D85\u51FA max_seq_len \u7684\u6837\u672C\u6BD4\u4F8B)",
    "vision token \u5747\u503C\u5728\u7406\u8BBA\u503C 20% \u4EE5\u5185",
    "\u5404\u7C7B\u9A8C\u8BC1\u901A\u8FC7\u7387 > 95%",
  ];
  s.addText(acceptance.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < acceptance.length - 1, fontSize: 10, color: C.dark_text },
  })), { x: 0.55, y: 4.95, w: 8.9, h: 0.45, fontFace: "Calibri", valign: "top", margin: 0 });

  addSlideNumber(s, 14, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 15 — Training Curriculum
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.off_white };
  addSectionTitle(s, "\u4E94\u9636\u6BB5\u8BFE\u7A0B\u5B66\u4E60 (Curriculum Training)");

  // Timeline bar
  const phaseColors = [C.deep, C.teal, C.accent_orange, C.accent_purple, C.light_teal];
  const phaseNames = ["Phase 1", "Phase 2", "C1", "C2", "Phase 5"];
  const phaseWidths = [1.6, 1.6, 2.2, 1.6, 1.6];
  let px = 0.6;
  phaseNames.forEach((p, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: 1.0, w: phaseWidths[i], h: 0.4, fill: { color: phaseColors[i] } });
    s.addText(p, { x: px, y: 1.0, w: phaseWidths[i], h: 0.4, fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    if (i < 4) {
      s.addText("\u25B6", { x: px + phaseWidths[i] - 0.05, y: 1.0, w: 0.2, h: 0.4, fontSize: 10, color: C.white, align: "center", valign: "middle", margin: 0 });
    }
    px += phaseWidths[i] + 0.05;
  });

  // Table
  s.addTable([
    // Header
    [
      { text: "\u9636\u6BB5", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10, align: "center" } },
      { text: "\u6570\u636E\u5185\u5BB9", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10 } },
      { text: "\u6837\u672C\u6570", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10, align: "center" } },
      { text: "\u5B66\u4E60\u76EE\u6807", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10 } },
      { text: "LR", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10, align: "center" } },
      { text: "Epochs", options: { fill: { color: C.midnight }, color: C.white, bold: true, fontSize: 10, align: "center" } },
    ],
    // P1
    [
      { text: "P1", options: { bold: true, color: C.deep, fontSize: 10, align: "center" } },
      { text: "silent + response\n\u57FA\u7840\u534F\u8BAE\u5BF9\u9F50", options: { fontSize: 9 } },
      { text: "~4K", options: { fontSize: 10, align: "center" } },
      { text: "\u5B66\u4F1A\u201C\u4F55\u65F6\u6C89\u9ED8\u3001\u4F55\u65F6\u56DE\u7B54\u201D", options: { fontSize: 9 } },
      { text: "1e-5", options: { fontSize: 10, align: "center" } },
      { text: "3", options: { fontSize: 10, align: "center" } },
    ],
    // P2
    [
      { text: "P2", options: { bold: true, color: C.teal, fontSize: 10, align: "center" } },
      { text: "+ recall + pending\n\u5F15\u5165\u68C0\u7D22\u4E0E\u5F85\u7B54", options: { fontSize: 9 } },
      { text: "~6K", options: { fontSize: 10, align: "center" } },
      { text: "\u5B66\u4F1A\u201C\u4F55\u65F6\u9700\u8981recall\u201D", options: { fontSize: 9 } },
      { text: "5e-6", options: { fontSize: 10, align: "center" } },
      { text: "3", options: { fontSize: 10, align: "center" } },
    ],
    // C1
    [
      { text: "C1", options: { bold: true, color: C.accent_orange, fontSize: 10, align: "center" } },
      { text: "+ compress w/ range\n\u5E26\u8303\u56F4\u6307\u5BFC\u7684\u538B\u7F29", options: { fontSize: 9 } },
      { text: "~15K", options: { fontSize: 10, align: "center" } },
      { text: "\u5B66\u4F1A\u201C\u5982\u4F55\u5199\u538B\u7F29\u6458\u8981\u201D", options: { fontSize: 9 } },
      { text: "3e-6", options: { fontSize: 10, align: "center" } },
      { text: "2", options: { fontSize: 10, align: "center" } },
    ],
    // C2
    [
      { text: "C2", options: { bold: true, color: C.accent_purple, fontSize: 10, align: "center" } },
      { text: "+ compress self-select\n\u81EA\u4E3B\u9009\u62E9\u538B\u7F29\u8303\u56F4", options: { fontSize: 9 } },
      { text: "~3K", options: { fontSize: 10, align: "center" } },
      { text: "\u5B66\u4F1A\u201C\u538B\u7F29\u54EA\u4E9B\u5185\u5BB9\u201D", options: { fontSize: 9 } },
      { text: "2e-6", options: { fontSize: 10, align: "center" } },
      { text: "2", options: { fontSize: 10, align: "center" } },
    ],
    // P5
    [
      { text: "P5", options: { bold: true, color: C.light_teal, fontSize: 10, align: "center" } },
      { text: "mixed, all types\n\u6DF7\u5408\u8BAD\u7EC3", options: { fontSize: 9 } },
      { text: "~5K", options: { fontSize: 10, align: "center" } },
      { text: "\u7EFC\u5408\u80FD\u529B\u878D\u5408", options: { fontSize: 9 } },
      { text: "1e-6", options: { fontSize: 10, align: "center" } },
      { text: "1", options: { fontSize: 10, align: "center" } },
    ],
  ], {
    x: 0.4, y: 1.6, w: 9.2, colW: [0.6, 2.2, 0.8, 2.6, 0.8, 0.8],
    border: { pt: 0.5, color: C.light_gray },
    fontFace: "Calibri",
    rowH: [0.35, 0.55, 0.55, 0.55, 0.55, 0.55],
    autoPage: false,
  });

  // Key notes
  addCard(s, 0.4, 4.85, 9.2, 0.65);
  const notes = [
    { text: "\u8BBE\u8BA1\u8981\u70B9\uFF1A", options: { bold: true, color: C.teal, fontSize: 11, breakLine: true } },
    { text: "\u2022 \u6BCF\u9636\u6BB5\u7EE7\u627F\u4E0A\u4E00\u9636\u6BB5 checkpoint\uFF0C\u800C\u975E\u4ECE base model \u91CD\u65B0\u5F00\u59CB", options: { fontSize: 10, color: C.dark_text, breakLine: true } },
    { text: "\u2022 Per-timestep: \u6BCF\u4E2A\u6837\u672C ~3,500 tokens \u2192 batch size 8-16 (vs \u4F20\u7EDF batch=4)", options: { fontSize: 10, color: C.dark_text, breakLine: true } },
    { text: "\u2022 Attention mask: causal + padding only\uFF0C\u65E0 sliding window\uFF08\u4EE5\u4FBF\u770B\u5230 recalled frames\uFF09", options: { fontSize: 10, color: C.dark_text } },
  ];
  s.addText(notes, { x: 0.55, y: 4.85, w: 8.9, h: 0.65, fontFace: "Calibri", valign: "middle", margin: 0, lineSpacingMultiple: 1.1 });

  addSlideNumber(s, 15, TOTAL);
}

// ════════════════════════════════════════════════════════════════
// SLIDE 16 — Summary & Future
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.midnight };

  // Decorative
  s.addShape(pres.shapes.OVAL, { x: -1, y: -1, w: 3, h: 3, fill: { color: C.deep, transparency: 60 } });
  s.addShape(pres.shapes.OVAL, { x: 8, y: 3.5, w: 4, h: 4, fill: { color: C.teal, transparency: 70 } });

  s.addText("ThinkStream: \u6838\u5FC3\u521B\u65B0\u603B\u7ED3", {
    x: 0.6, y: 0.25, w: 8.8, h: 0.55,
    fontSize: 26, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.8, w: 1.2, h: 0.04, fill: { color: C.light_teal } });

  // 3x2 grid of innovations
  const innovations = [
    { title: "Per-Timestep\nRe-render", desc: "\u6D88\u9664 KV \u7D2F\u79EF\uFF0C\u4FDD\u8BC1\u8BAD\u7EC3/\u63A8\u7406\u4E00\u81F4\u6027" },
    { title: "RCSM\n\u538B\u7F29\u8BB0\u5FC6", desc: "\u56FA\u5B9A token \u9884\u7B97\uFF0C\u652F\u6301\u65E0\u9650\u957F\u5EA6\u89C6\u9891" },
    { title: "\u591A\u7EF4\u5EA6\n\u538B\u7F29\u8BC4\u5206", desc: "\u4F18\u5316\u672A\u6765\u53EF\u56DE\u7B54\u6027\uFF0C\u800C\u975E\u7B80\u5355\u4FE1\u606F\u635F\u5931" },
    { title: "Temporal\nMROPE", desc: "\u8DE8\u6A21\u6001\u65F6\u95F4\u5BF9\u9F50\u6CE8\u610F\u529B\u589E\u5F3A" },
    { title: "Zone-Based\nKV Cache", desc: "90% prefill \u51CF\u5C11\uFF0C\u5B9E\u65F6\u63A8\u7406\u53EF\u884C" },
    { title: "Question-Blind\nPipeline", desc: "\u6770\u7EDD\u672A\u6765\u4FE1\u606F\u6CC4\u6F0F\uFF0C\u8BAD\u7EC3\u6570\u636E\u7EAF\u51C0" },
  ];

  innovations.forEach((inv, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const ix = 0.5 + col * 3.15;
    const iy = 1.1 + row * 1.55;

    s.addShape(pres.shapes.RECTANGLE, {
      x: ix, y: iy, w: 2.85, h: 1.3,
      fill: { color: C.midnight },
      line: { color: C.light_teal, width: 1.2 },
    });
    s.addText(inv.title, { x: ix + 0.1, y: iy + 0.08, w: 2.65, h: 0.55, fontSize: 13, fontFace: "Calibri", color: C.ice, bold: true, margin: 0, lineSpacingMultiple: 1.1 });
    s.addText(inv.desc, { x: ix + 0.1, y: iy + 0.65, w: 2.65, h: 0.55, fontSize: 10, fontFace: "Calibri", color: C.sub_text, margin: 0, lineSpacingMultiple: 1.2 });
  });

  // Next Steps
  s.addText("Next Steps", {
    x: 0.6, y: 4.25, w: 2, h: 0.35, fontSize: 16, fontFace: "Georgia", color: C.light_teal, bold: true, margin: 0,
  });

  const nextSteps = [
    { x: 0.6, label: "Phase 1\nTraining", color: C.deep },
    { x: 2.8, label: "DAgger\nLoop", color: C.teal },
    { x: 5.0, label: "RL Training\n(GRPO)", color: C.light_teal },
  ];
  nextSteps.forEach((ns, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: ns.x, y: 4.7, w: 1.8, h: 0.55, fill: { color: ns.color, transparency: 50 }, line: { color: ns.color, width: 1 } });
    s.addText(ns.label, { x: ns.x, y: 4.7, w: 1.8, h: 0.55, fontSize: 10, fontFace: "Calibri", color: C.white, align: "center", valign: "middle", margin: 0 });
    if (i < 2) addArrowRight(s, ns.x + 1.8, 4.97, 0.8);
  });

  // Footer
  s.addText("arXiv: 2603.12938  |  Thinking in Streaming Video  |  2026", {
    x: 0.6, y: 5.3, w: 8, h: 0.25,
    fontSize: 9, fontFace: "Calibri", color: C.sub_text,
  });

  addSlideNumber(s, 16, TOTAL);
}

// ─── Generate ───
pres.writeFile({ fileName: "/Users/hzh/Desktop/简历/streamvideo-project/ThinkStream/ThinkStream_Presentation.pptx" })
  .then(() => console.log("DONE: ThinkStream_Presentation.pptx"))
  .catch(err => console.error("ERROR:", err));
