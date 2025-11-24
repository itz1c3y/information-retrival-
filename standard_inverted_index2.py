

import re
import os
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox

# ---------------------------
# Documents from disk (PDF version)
# ---------------------------
import PyPDF2
doc_folder = r"C:\Users\Sahraee\Downloads\Documents"
documents = {}

for i in range(1, 4):
    path = os.path.join(doc_folder, f"{i}.pdf")  # now looks for 1.pdf, 2.pdf, 3.pdf
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document {i} not found at {path}")
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        documents[i] = text


# ---------------------------
# Stopwords & Tokenization
# ---------------------------
STOPWORDS = set([
    'a','an','the','and','or','not','in','on','at','for','to','of','is','are','was','were','be','by','with','as','that','this','these','those','it'
])
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def normalize(text):
    return text.lower()

def tokenize(text):
    return TOKEN_RE.findall(normalize(text))

# ---------------------------
# Porter stemmer 
# ---------------------------
try:
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    def stem(w):
        return porter.stem(w)
except Exception:
    def stem(w):
        suffixes = ['ing','ly','ed','s','es','er']
        for suf in suffixes:
            if w.endswith(suf) and len(w) > len(suf)+2:
                return w[:-len(suf)]
        return w

# ---------------------------
#  B-Tree implementation
# ---------------------------
class BTreeNode:
    def __init__(self, t, leaf=True):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

class BTree:
    def __init__(self, t=2):
        self.t = t
        self.root = BTreeNode(t)

    def search(self, k, node=None):
        if node is None:
            node = self.root
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:
            return True
        if node.leaf:
            return False
        return self.search(k, node.children[i])

    def split_child(self, parent, idx):
        t = self.t
        node = parent.children[idx]
        new_node = BTreeNode(t, node.leaf)
        parent.keys.insert(idx, node.keys[t-1])
        parent.children.insert(idx+1, new_node)
        new_node.keys = node.keys[t:]
        node.keys = node.keys[:t-1]
        if not node.leaf:
            new_node.children = node.children[t:]
            node.children = node.children[:t]

    def insert(self, k):
        root = self.root
        if len(root.keys) == 2*self.t - 1:
            new_root = BTreeNode(self.t, False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root
            self._insert_nonfull(new_root, k)
        else:
            self._insert_nonfull(root, k)

    def _insert_nonfull(self, node, k):
        if node.leaf:
            node.keys.append(k)
            node.keys.sort()
        else:
            i = len(node.keys) - 1
            while i >= 0 and k < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2*self.t - 1:
                self.split_child(node, i)
                if k > node.keys[i]:
                    i += 1
            self._insert_nonfull(node.children[i], k)

# ---------------------------
# Build SID table 
# ---------------------------
inverted_index = defaultdict(lambda: defaultdict(int))
dict_btree = BTree(t=2)
process_trace = []

def log(step, detail):
    process_trace.append({"step": step, "detail": detail})

for doc_id, text in documents.items():
    log("DOCUMENT", f"Doc {doc_id}: {text[:100]}...")  # first 100 chars
    norm = normalize(text)
    log("NORMALIZE", f"Doc {doc_id}: {norm[:100]}...")
    toks = tokenize(norm)
    log("TOKENIZE", f"Doc {doc_id}: {toks[:20]}...")  # first 20 tokens

    for tok in toks:
        if tok in STOPWORDS:
            log("STOPWORD_REMOVE", f"REMOVED: {tok}")
            continue
        st = stem(tok)
        log("STEM", f"{tok} -> {st}")
        inverted_index[st][doc_id] += 1
        log("SID_INSERT", f"Term '{st}' occurs in doc {doc_id}")
        if not dict_btree.search(st):
            dict_btree.insert(st)
            log("BTREE_INSERT", f"Inserted '{st}' into B-Tree")

# ---------------------------
# Print SID 
# ---------------------------
def print_sid_table(inverted_index):
    print("\n=== Standard Inverted Index (One Term Per Line) ===\n")
    for term in sorted(inverted_index.keys()):
        postings = {int(k): v for k, v in inverted_index[term].items()}
        print(f"{term:<15} -> {postings}")

print_sid_table(inverted_index)

# ---------------------------
# Save JSON files
# ---------------------------
os.makedirs('output', exist_ok=True)
import json
with open('output/sid.json', 'w', encoding='utf-8') as f:
    json.dump({k: dict(v) for k,v in inverted_index.items()}, f, ensure_ascii=False, indent=2)
with open('output/process_trace.json', 'w', encoding='utf-8') as f:
    json.dump(process_trace, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------
#  B‑TREE VISUALISER
# --------------------------------------------------------------
class TreeVisualizer:
    def __init__(self, btree, postings):
        self.btree   = btree
        self.postings = postings
        self.pos     = {}
        self.level_y = 130
        self.w = 110               # fixed node width
        self.h = 45

        # ---- window ------------------------------------------------
        self.root = tk.Tk()
        self.root.title('Compact B‑Tree')
        self.canvas = tk.Canvas(
            self.root,
            width=1400, height=800,
            bg='white',
            scrollregion=(0,0,3000,2000)
        )
        hbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL,
                            command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(self.root, orient=tk.VERTICAL,
                            command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(xscrollcommand=hbar.set,
                           yscrollcommand=vbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ---- prune single‑term leaves -------------------------------
        self._prune(self.btree.root)

        # ---- layout ------------------------------------------------
        self._calc_width(self.btree.root)
        self._place(self.btree.root, 0)
        self._draw()
        self.root.mainloop()

    # ----------------------------------------------------------
    def _prune(self, node):
        """Remove leaf nodes that hold only one key."""
        if node.leaf:
            if len(node.keys) == 1:
                pass
            return
        for c in node.children:
            self._prune(c)

    # ----------------------------------------------------------
    def _calc_width(self, node):
        if node.leaf:
            self.pos[id(node)] = {'width': 1}
            return
        total = 0
        for c in node.children:
            self._calc_width(c)
            total += self.pos[id(c)]['width']
        self.pos[id(node)] = {'width': max(total, 1)}

    # ----------------------------------------------------------
    def _place(self, node, start):
        if node.leaf:
            self.pos[id(node)]['x'] = start
            self.pos[id(node)]['d'] = self._depth(node)
            return start + 1

        cur = start
        for c in node.children:
            cur = self._place(c, cur)

        xs = [self.pos[id(c)]['x'] for c in node.children]
        center = (xs[0] + xs[-1]) / 2 if xs else start
        self.pos[id(node)]['x'] = center
        self.pos[id(node)]['d'] = self._depth(node)
        return max(cur, start + 1)

    # ----------------------------------------------------------
    def _depth(self, node):
        d = 0
        cur = node
        while hasattr(cur, 'children') and cur.children:
            cur = cur.children[0]
            d += 1
        return d

    # ----------------------------------------------------------
    def _draw(self):
        self._draw_edges(self.btree.root)
        self._draw_nodes(self.btree.root)

    def _draw_edges(self, node):
        if not node.children:
            return
        px, py = self._pixel(node)
        for c in node.children:
            cx, cy = self._pixel(c)
            self.canvas.create_line(
                px, py + self.h//2,
                cx, cy - self.h//2,
                fill='gray', width=1
            )
            self._draw_edges(c)

    def _draw_nodes(self, node):
        x, y = self._pixel(node)

        txt = node.keys[0] if node.keys else ''
        if len(node.keys) > 1:
            txt += ', …'

        rect = self.canvas.create_rectangle(
            x-self.w//2, y-self.h//2,
            x+self.w//2, y+self.h//2,
            fill='#e1f5fe', outline='#0288d1', width=2
        )
        txt_id = self.canvas.create_text(
            x, y, text=txt, font=('Consolas', 10, 'bold')
        )

        def on_click(event, keys=node.keys):
            lines = []
            for k in keys:
                p = dict(self.postings.get(k, {}))
                lines.append(f"{k}: {p}" if p else f"{k}: (no postings)")
            messagebox.showinfo("Postings", "\n".join(lines) if lines else "No data")
        self.canvas.tag_bind(rect,   "<Button-1>", on_click)
        self.canvas.tag_bind(txt_id, "<Button-1>", on_click)

        for c in node.children:
            self._draw_nodes(c)

    # ----------------------------------------------------------
    def _pixel(self, node):
        info = self.pos[id(node)]
        ux   = info['x']
        depth = info['d']

        max_u = max(p['x'] for p in self.pos.values())
        cw    = max(self.canvas.winfo_width(), 1400)
        margin = 100
        avail = cw - 2*margin
        px = margin + ux * (avail / max(1, max_u))
        py = 80 + depth * self.level_y
        return px, py
# ---------------------------
# Run 
# ---------------------------
TreeVisualizer(dict_btree, inverted_index)
