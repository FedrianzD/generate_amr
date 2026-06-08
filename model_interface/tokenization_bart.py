# coding:utf-8
# this is a simplified version of "https://github.com/SapienzaNLP/spring/blob/main/spring_amr/tokenization_bart.py"
import sys
import penman
import itertools
import regex as re
import traceback

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

from .tokenization_mbart50 import MBart50Tokenizer
from common import postprocessing
from common.penman_interface import encode
from common.constant import raw_special_tokens, recategorizations


class AMRBartTokenizer(MBart50Tokenizer):
    INIT = 'Ġ'

    def __init__(self,
                 vocab_file,
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 mask_token="<mask>",
                 **kwargs):
        super().__init__(vocab_file,
                         #  bos_token=bos_token,
                         eos_token=eos_token,
                         sep_token=sep_token,
                         cls_token=cls_token,
                         unk_token=unk_token,
                         pad_token=pad_token,
                         mask_token=mask_token,
                         #  add_prefix_space=add_prefix_space,
                         **kwargs)
        self.modified = 0
        self.recategorizations = set(recategorizations)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.remove_pars = False

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary()
        return inst

    def init_amr_vocabulary(self):
        #~ Compatibilities from the original AMRBartTokenizer
        self.vocab = self.get_vocab()
        tokens = [t for t in raw_special_tokens if t not in self.vocab]
        super().add_tokens(tokens)
        self.modified = len(tokens)
        self.vocab = self.get_vocab()
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        #~

        self.amr_bos_token = "<AMR>"
        self.amr_bos_token_id = self.vocab[self.amr_bos_token]
        self.amr_eos_token = "</AMR>"
        self.amr_eos_token_id = self.vocab[self.amr_eos_token]
        print(f"Added {self.modified} AMR tokens")

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.vocab:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                return super()._tokenize(text)

        return bpe_tokens

    def _tok_bpe(self, token):
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.vocab:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            #! Not sure.
            return super()._tokenize(token)
            # for tok in self.patterns.findall(' ' + token):
            #     tok = "".join(
            #         self.byte_encoder[b] for b in tok.encode("utf-8"))
            #     toks = self.tokenize(tok)
            #     tokk.extend(toks)
        return tokk

    def tokenize_amr(self, amr_tokens):
        bpe_tokens = []
        for i, tokk in enumerate(amr_tokens):
            is_in_enc = self.INIT + tokk in self.vocab
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            # dealing with examples like "The_United_Kingdom_of_xxx"
            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + "<lit>"]
                bpe_toks += self._tok_bpe(tokk)
                bpe_toks.append(self.INIT + "</lit>")

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3]) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.vocab:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + \
                            self._tok_bpe(rel[1:]) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:])
                else:
                    print("tok:", tokk)
                    print(
                        f"is_rel:{is_rel}, is_spc:{is_spc}, is_frame:{is_frame}, is_of:{is_of}")
                    exit()
                    raise
            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk)

            bpe_tokens.append(bpe_toks)
        bpe_tokens = list(itertools.chain.from_iterable(bpe_tokens))
        bpe_token_ids = [self.vocab.get(b, self.unk_token_id) for b in bpe_tokens]
        return bpe_token_ids

    def decode_amr(self, tokens, restore_name_ops=None, prefix="unk"):
        # print()
        # print()
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print()
        # print()
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(
                tokens, self)
        except Exception as e:
            print('Decoding failure', file=sys.stderr)
            print(get_traceback(e), file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph = self._fix_and_make_graph(nodes)
            # if collapse_name_ops:
            #     graph = postprocessing._split_name_ops(graph)
        except Exception as e:
            print('Building failure', file=sys.stderr)
            print(get_traceback(e), file=sys.stderr)
            # print('nodes', nodes, file=sys.stderr)
            # print('backreferences', backreferences, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                # print('nodes', nodes, file=sys.stderr)
                # print('backreferences', backreferences, file=sys.stderr)
                # print('graph', graph, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnection 2 failure', file=sys.stderr)
            print(get_traceback(e), file=sys.stderr)
            # print('nodes', nodes, file=sys.stderr)
            # print('backreferences', backreferences, file=sys.stderr)
            # print('graph', graph, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)

    def _fix_and_make_graph(self, nodes):

        # print(">>>>>>>> nodes1: ", nodes, "\n")

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        # print(">>>>>>>> nodes2: ", nodes, "\n")

        i = 0
        nodes_ = []
        while i < len(nodes):
            nxt = nodes[i]
            pst = None
            if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                e = nxt.find('>')
                if e != len(nxt) - 1:
                    pst = nxt[e+1:]
                    nxt = nxt[:e+1]
                nodes_.append(nxt)
                if pst is not None:
                    nodes_.append(pst)
            else:
                nodes_.append(nxt)
            i += 1
        nodes = nodes_

        # print(">>>>>>>> nodes2.5: ", nodes, "\n")

        # i = 1
        # nodes_ = [nodes[0]]
        i = 0
        nodes_ = []
        while i < len(nodes):
            nxt = nodes[i]
            if (i < len(nodes) - 1) and isinstance(nxt, str) and nxt.startswith('<pointer:'):
                nxt = 'z' + nxt[9:-1]
                fol = nodes[i+1]
                # is not expansion
                if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                    nodes_.append(nxt)
                else:
                    if self.remove_pars:
                        nodes_.append('(')
                    else:
                        if len(nodes_) == 0 or nodes_[-1] != '(':
                            nodes_.append('(')
                            # pass
                    nodes_.append(nxt)
                    nodes_.append('/')
            else:
                nodes_.append(nxt)
            i += 1
        nodes = nodes_

        # print(">>>>>>>> nodes3: ", nodes, "\n")

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ':':
                nodes_.append(nodes[i] + nodes[i+1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        # print(">>>>>>>> nodes4: ", nodes, "\n")

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == '/' and nodes[i] == '/':
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        # print(">>>>>>>> nodes5: ", nodes, "\n")

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == '/':
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_

        # print(">>>>>>>> nodes6: ", nodes, "\n")


        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in nodes:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

        # print(">>>>>>>> nodes7: ", nodes, "\n")

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append('(')
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', '') + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == 'CONST':
                    quote = False
                    for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if prev == '(':
                    if next in ('VAR', 'I'):
                        pieces.append(piece)
                elif prev == ')':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'VAR':
                    if next in ('/', 'EDGE', 'MODE', ')'):
                        pieces.append(piece)
                elif prev == '/':
                    if next in ('INST', 'I'):
                        pieces.append(piece)
                elif prev == 'INST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'I':
                    if next in ('/', ')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'EDGE':
                    if next in ('(', 'VAR', 'CONST', 'I'):
                        pieces.append(piece)
                    elif next == ')':
                        pieces[-1] = piece
                    elif next in ('EDGE', 'MODE'):
                        pieces[-1] = piece
                elif prev == 'MODE':
                    if next == 'INST':
                        pieces.append(piece)
                elif prev == 'CONST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)

        # print(">>>>>>>> pieces1: ", pieces, "\n")

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in pieces:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

        # print(">>>>>>>> pieces2: ", pieces, "\n")

        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + ' ')
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ':instance' and y is None:
                triples.append(penman.Triple(x, rel, 'thing'))
            elif y is None:
                var = f'z{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0

            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + \
                    ' / ' + match.group(2) + match.group(3)
                n += 1
                return out
            linearized = re.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            def _repl2(match):
                return match.group(1)
            linearized = re.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                                linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            # adds a ':' to args w/o it
            linearized = re.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)
        # print(">>>>>>>> linearized: ", linearized, "\n")
        g = penman.decode(linearized)
        return g

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == 'i':
            return "I"
        elif re.match(r'^[a-z]\d*$', node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ('+', '-'):
            return "CONST"
        elif node == ':mode':
            return 'MODE'
        elif node.startswith(':'):
            return "EDGE"
        elif node in ['/', '(', ')']:
            return node
        elif node[0].isalpha():
            for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return 'CONST'


from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
