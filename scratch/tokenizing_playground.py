
tok = Mecab()

allowed_pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')

texts = corpus_agg['text']
texts = [re.sub(r"([.?!])\1+", r'\1', line) for line in texts]
texts = [allowed_pattern.sub(' ', line).strip() for line in texts]
tokens_matrix = [[token[0] for token in tok.pos(line) if token[1] in allowed_pos] for line in texts]
