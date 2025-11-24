

import re
import json
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict

# ---------------------------
# Documents
# ---------------------------
documents = {
    1: "Python is a great programming language. Programmers use Python for data science.",
    2: "Information Retrieval involves indexing documents. An inverted index speeds up search.",
    3: "Machine learning uses data. Python and algorithms help analyze this data efficiently."
}

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
# Porter stemmer (prefer NLTK if available)
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

    def to_dict(self, node=None):
        if node is None:
            node = self.root
        return {
            'keys': node.keys,
            'leaf': node.leaf,
            'children': [self.to_dict(c) for c in node.children]
        }

# ---------------------------
# Build SID table
# ---------------------------
inverted_index = defaultdict(lambda: defaultdict(int))
dict_btree = BTree(t=2)

for doc_id, text in documents.items():
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        s = stem(tok)
        inverted_index[s][doc_id] += 1
        if not dict_btree.search(s):
            dict_btree.insert(s)

# ---------------------------
# PRINT SID TABLE 
# ---------------------------
def print_sid_table(inverted_index):
    print("\n=== Standard Inverted Index (One Term Per Line) ===\n")
    for term in sorted(inverted_index.keys()):
        postings = {int(k): v for k, v in inverted_index[term].items()}
        print(f"{term:<12} -> {postings}")

print_sid_table(inverted_index)
# ---------------------------
# Tkinter visualizer
# ---------------------------
class TreeVisualizer:
    def __init__(self, btree, postings):
        self.btree = btree
        self.postings = postings
        self.node_positions = {}  # node -> (x,y)
        self.node_sizes = {}
        self.canvas_nodes = []

        self.counter_x = 0
        self.level_y = 80
        self.node_width = 100
        self.node_height = 30

        self.root = tk.Tk()
        self.root.title('B-Tree Visualizer')
        self.canvas = tk.Canvas(self.root, width=1200, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.on_resize)

        self.compute_positions(self.btree.root, depth=0)
        self.draw_tree()

    def compute_positions(self, node, depth=0):
        if node.leaf:
            x = self.counter_x
            self.counter_x += 1
            self.node_positions[id(node)] = (x, depth)
        else:
            for i, child in enumerate(node.children):
                self.compute_positions(child, depth+1)
            xs = [self.node_positions[id(c)][0] for c in node.children]
            avg_x = sum(xs)/len(xs)
            self.node_positions[id(node)] = (avg_x, depth)

    def draw_tree(self):
        if not self.node_positions:
            return
        xs = [pos[0] for pos in self.node_positions.values()]
        min_x, max_x = min(xs), max(xs)
        span = max_x - min_x if max_x > min_x else 1
        width = int(self.canvas.winfo_width())
        height = int(self.canvas.winfo_height())
        margin = 50

        def to_pixel(x, depth):
            px = margin + ((x - min_x) / span) * (width - 2*margin)
            py = margin + depth * self.level_y
            return px, py

        self.canvas.delete('all')
        self.draw_links(self.btree.root, to_pixel)
        self.draw_nodes(self.btree.root, to_pixel)

    def draw_links(self, node, to_pixel):
        node_id = id(node)
        x_log, depth = self.node_positions[node_id]
        x_px, y_px = to_pixel(x_log, depth)
        for child in node.children:
            cx_log, cdepth = self.node_positions[id(child)]
            cx_px, cy_px = to_pixel(cx_log, cdepth)
            self.canvas.create_line(x_px, y_px + 15, cx_px, cy_px - 15)
            self.draw_links(child, to_pixel)

    def draw_nodes(self, node, to_pixel):
        node_id = id(node)
        x_log, depth = self.node_positions[node_id]
        x_px, y_px = to_pixel(x_log, depth)
        keys_text = ','.join(node.keys)
        w = max(self.node_width, 8 * (len(keys_text)+1))
        h = self.node_height
        left = x_px - w/2
        top = y_px - h/2
        rect = self.canvas.create_rectangle(left, top, left+w, top+h, fill='#f0f0ff', outline='#000')
        text = self.canvas.create_text(x_px, y_px, text=keys_text)
        self.canvas.tag_bind(rect, '<Button-1>', lambda e, n=node: self.on_node_click(n))
        self.canvas.tag_bind(text, '<Button-1>', lambda e, n=node: self.on_node_click(n))
        for child in node.children:
            self.draw_nodes(child, to_pixel)

    def on_node_click(self, node):
        terms = node.keys
        lines = []
        for t in terms:
            postings = self.postings.get(t, {})
            lines.append(f"{t} -> {dict(postings)}")
        if not lines:
            messagebox.showinfo('Node Postings', 'No terms')
        else:
            messagebox.showinfo('Node Postings', ''.join(lines))

    def on_resize(self, event):
        self.draw_tree()

    def run(self):
        self.root.mainloop()

# ---------------------------
# Run visualizer
# ---------------------------
if __name__ == '__main__':
    # save sid for inspection
    with open('sid.json', 'w', encoding='utf-8') as f:
        json.dump({k: dict(v) for k,v in inverted_index.items()}, f, ensure_ascii=False, indent=2)

    visualizer = TreeVisualizer(dict_btree, inverted_index)
    visualizer.run()

process_trace = []

def log(step, detail):
    process_trace.append({"step": step, "detail": detail})

inverted_index = defaultdict(lambda: defaultdict(int))
dict_btree = BTree(t=2)
process_trace.clear()

for doc_id, text in documents.items():
    log("DOCUMENT", f"Doc {doc_id}: {text}")
    norm = normalize(text)
    log("NORMALIZE", f"Doc {doc_id}: {norm}")
    toks = tokenize(norm)
    log("TOKENIZE", f"Doc {doc_id}: {toks}")

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

with open('sid.json', 'w', encoding='utf-8') as f:
    json.dump({k: dict(v) for k,v in inverted_index.items()}, f, ensure_ascii=False, indent=2)

with open('process_trace.json', 'w', encoding='utf-8') as f:
    json.dump(process_trace, f, ensure_ascii=False, indent=2)
