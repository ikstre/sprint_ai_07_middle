"""
Generate simple PNG architecture diagrams using only the Python standard library.

Outputs:
- docs/diagram_architecture.png
- docs/diagram_paths.png
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

WHITE = (250, 250, 248)
BLACK = (25, 25, 25)
GRAY = (100, 104, 110)
LIGHT_GRAY = (230, 232, 235)
GREEN = (205, 236, 211)
GREEN_BORDER = (92, 153, 103)
BLUE = (209, 228, 255)
BLUE_BORDER = (79, 125, 201)
YELLOW = (247, 235, 186)
YELLOW_BORDER = (184, 150, 43)
RED = (245, 214, 214)
RED_BORDER = (181, 85, 85)


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "*": ["00000", "01010", "00100", "11111", "00100", "01010", "00000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = WHITE):
        self.width = width
        self.height = height
        self.pixels = [[bg for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for yy in range(y, y + h):
            if 0 <= yy < self.height:
                row = self.pixels[yy]
                for xx in range(max(0, x), min(self.width, x + w)):
                    row[xx] = color

    def rect(self, x: int, y: int, w: int, h: int, fill: tuple[int, int, int], border: tuple[int, int, int], border_w: int = 2) -> None:
        self.fill_rect(x, y, w, h, fill)
        for i in range(border_w):
            self.line(x + i, y + i, x + w - 1 - i, y + i, border)
            self.line(x + i, y + h - 1 - i, x + w - 1 - i, y + h - 1 - i, border)
            self.line(x + i, y + i, x + i, y + h - 1 - i, border)
            self.line(x + w - 1 - i, y + i, x + w - 1 - i, y + h - 1 - i, border)

    def line(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], thickness: int = 2) -> None:
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self.set_pixel(x1, y1, color)
            return
        for i in range(steps + 1):
            x = round(x1 + dx * i / steps)
            y = round(y1 + dy * i / steps)
            for ox in range(-(thickness // 2), thickness // 2 + 1):
                for oy in range(-(thickness // 2), thickness // 2 + 1):
                    self.set_pixel(x + ox, y + oy, color)

    def arrow(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        self.line(x1, y1, x2, y2, color, 2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        left = angle + math.pi * 0.84
        right = angle - math.pi * 0.84
        lx = int(x2 + size * math.cos(left))
        ly = int(y2 + size * math.sin(left))
        rx = int(x2 + size * math.cos(right))
        ry = int(y2 + size * math.sin(right))
        self.line(x2, y2, lx, ly, color, 2)
        self.line(x2, y2, rx, ry, color, 2)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int] = BLACK, scale: int = 2) -> None:
        cursor = x
        for ch in text.upper():
            glyph = FONT.get(ch, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor + gx * scale, y + gy * scale, scale, scale, color)
            cursor += (len(glyph[0]) + 1) * scale

    def centered_text(self, x: int, y: int, w: int, h: int, text: str, color: tuple[int, int, int] = BLACK, scale: int = 2) -> None:
        lines = text.split("\n")
        glyph_w = 6 * scale
        glyph_h = 7 * scale
        total_h = len(lines) * glyph_h + (len(lines) - 1) * scale * 2
        start_y = y + max(0, (h - total_h) // 2)
        for idx, line in enumerate(lines):
            line_w = max(0, len(line) * glyph_w - scale)
            start_x = x + max(0, (w - line_w) // 2)
            self.text(start_x, start_y + idx * (glyph_h + scale * 2), line, color, scale)

    def save_png(self, path: Path) -> None:
        raw = bytearray()
        for row in self.pixels:
            raw.append(0)
            for r, g, b in row:
                raw.extend([r, g, b])

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", ihdr)
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def box(c: Canvas, x: int, y: int, w: int, h: int, label: str, fill: tuple[int, int, int], border: tuple[int, int, int]) -> None:
    c.rect(x, y, w, h, fill, border, 2)
    c.centered_text(x, y, w, h, label, BLACK, 2)


def architecture_diagram() -> None:
    c = Canvas(1800, 1100)
    c.centered_text(650, 30, 500, 40, "PROJECT STRUCTURE", BLACK, 3)

    # Large lanes
    c.rect(120, 120, 620, 840, YELLOW, YELLOW_BORDER, 2)
    c.rect(1060, 120, 620, 840, BLUE, BLUE_BORDER, 2)
    c.rect(760, 160, 280, 780, GREEN, GREEN_BORDER, 2)
    # Lane title bars (avoid center-overlap text artifacts)
    box(c, 220, 140, 420, 52, "A PATH / AUTORAG", LIGHT_GRAY, YELLOW_BORDER)
    box(c, 1140, 140, 460, 52, "B PATH / SERVICE RAG", LIGHT_GRAY, BLUE_BORDER)
    box(c, 790, 178, 220, 48, "COMMON MODULES", LIGHT_GRAY, GREEN_BORDER)

    common = [
        (800, 240, "CONFIG"),
        (800, 340, "LOADER"),
        (800, 440, "CHUNKER"),
        (800, 540, "VECTORSTORE"),
        (800, 640, "RETRIEVER"),
        (800, 740, "GENERATOR"),
        (800, 840, "RAGPIPELINE"),
    ]
    for x, y, label in common:
        box(c, x, y, 200, 60, label, LIGHT_GRAY, GRAY)

    a_nodes = [
        (190, 220, 220, 70, "RUN_PIPELINE"),
        (190, 350, 220, 70, "PREPARE\nAUTORAG CSV"),
        (190, 490, 220, 70, "FINETUNE\nLOCAL"),
        (190, 630, 220, 70, "RUN_AUTORAG\nOPT"),
        (190, 780, 220, 70, "BEST\nCONDITION"),
        (450, 350, 220, 70, "CORPUS.PARQUET\nQA.PARQUET"),
        (450, 510, 220, 70, "MODELS\nFINETUNED"),
        (450, 690, 220, 70, "AUTORAG\nRESULTS"),
    ]
    for x, y, w, h, label in a_nodes:
        box(c, x, y, w, h, label, LIGHT_GRAY, GRAY)

    b_nodes = [
        (1130, 220, 220, 70, "INDEX\nDOCUMENTS"),
        (1130, 360, 220, 70, "APP.PY"),
        (1130, 500, 220, 70, "RUN\nEVALUATION"),
        (1130, 640, 220, 70, "CHECK\nRELEASE GATE"),
        (1390, 260, 220, 70, "VECTOR DB"),
        (1390, 430, 220, 70, "USER\nRESPONSE"),
        (1390, 600, 220, 70, "EVAL\nREPORT"),
    ]
    for x, y, w, h, label in b_nodes:
        box(c, x, y, w, h, label, LIGHT_GRAY, GRAY)

    # A arrows
    c.arrow(300, 290, 300, 350, BLACK)
    c.arrow(300, 420, 300, 490, BLACK)
    c.arrow(300, 560, 300, 630, BLACK)
    c.arrow(300, 700, 300, 780, BLACK)
    c.arrow(410, 385, 450, 385, BLACK)
    c.arrow(410, 525, 450, 545, BLACK)
    c.arrow(410, 665, 450, 725, BLACK)

    # B arrows
    c.arrow(1240, 290, 1240, 360, BLACK)
    c.arrow(1240, 430, 1240, 500, BLACK)
    c.arrow(1240, 570, 1240, 640, BLACK)
    c.arrow(1350, 255, 1390, 295, BLACK)
    c.arrow(1350, 395, 1390, 465, BLACK)
    c.arrow(1350, 535, 1390, 635, BLACK)

    # Common connections
    for y in [270, 370, 470, 570, 670, 770, 870]:
        c.line(740, y, 800, y, GREEN_BORDER, 2)
        c.line(1000, y, 1060, y, GREEN_BORDER, 2)

    c.save_png(DOCS / "diagram_architecture.png")


def path_diagram() -> None:
    c = Canvas(1800, 980)
    c.centered_text(560, 26, 700, 40, "A AND B EXECUTION PATHS", BLACK, 3)

    # Top B path
    top_y = 120
    labels = [
        ("GOAL", 40, 120),
        ("B PATH", 250, 140),
        ("DOC LOADER", 470, 160),
        ("CHUNKER", 700, 140),
        ("INDEX /\nVECTORSTORE", 920, 200),
        ("RETRIEVER", 1180, 160),
        ("GENERATOR", 1390, 160),
        ("RESPONSE /\nEVAL", 1600, 160),
    ]
    prev_x = None
    prev_w = None
    for label, x, w in labels:
        box(c, x, top_y, w, 84, label, GREEN, GREEN_BORDER)
        if prev_x is not None:
            c.arrow(prev_x + prev_w, top_y + 42, x, top_y + 42, BLACK)
        prev_x = x
        prev_w = w

    # Bottom A path
    box(c, 260, 360, 180, 84, "A PATH", YELLOW, YELLOW_BORDER)
    box(c, 520, 320, 240, 84, "RUN_PIPELINE", YELLOW, YELLOW_BORDER)
    box(c, 860, 210, 260, 84, "PREPARE\nAUTORAG CSV", LIGHT_GRAY, GRAY)
    box(c, 860, 360, 260, 84, "FINETUNE\nLOCAL", LIGHT_GRAY, GRAY)
    box(c, 860, 510, 260, 84, "RUN_AUTORAG\nOPT", LIGHT_GRAY, GRAY)
    box(c, 1260, 230, 250, 70, "CORPUS / QA", BLUE, BLUE_BORDER)
    box(c, 1260, 380, 250, 70, "FINETUNED\nMODELS", BLUE, BLUE_BORDER)
    box(c, 1260, 530, 250, 70, "AUTORAG\nRESULTS", BLUE, BLUE_BORDER)
    box(c, 1570, 510, 180, 84, "BEST\nSETTING", RED, RED_BORDER)

    c.arrow(440, 402, 520, 362, BLACK)
    c.arrow(760, 362, 860, 252, BLACK)
    c.arrow(760, 362, 860, 402, BLACK)
    c.arrow(760, 362, 860, 552, BLACK)
    c.arrow(1120, 252, 1260, 265, BLACK)
    c.arrow(1120, 402, 1260, 415, BLACK)
    c.arrow(1120, 552, 1260, 565, BLACK)
    c.arrow(1510, 565, 1570, 552, BLACK)

    c.centered_text(260, 700, 1280, 36, "A PATH USES RUN_PIPELINE AS THE FINAL ENTRYPOINT", BLACK, 2)
    c.centered_text(260, 740, 1280, 36, "B PATH USES INDEX_DOCUMENTS.PY -> APP.PY -> EVALUATION", BLACK, 2)
    c.centered_text(260, 780, 1280, 36, "A RESULTS AND B GATE REPORTS SHOULD NOT BE READ AS THE SAME AXIS", RED_BORDER, 2)

    c.save_png(DOCS / "diagram_paths.png")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    architecture_diagram()
    path_diagram()
    print(DOCS / "diagram_architecture.png")
    print(DOCS / "diagram_paths.png")


if __name__ == "__main__":
    main()
