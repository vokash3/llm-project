from BPE import BPE

if __name__ == '__main__':
    vocab_size: int = 10
    text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'

    bpe = BPE(vocab_size=vocab_size)
    bpe.fit(text)  # делаем токены из текста
    print("TOKEN TO ID:")
    for id, token in enumerate(bpe.token2id):
        print(f"{id}: {token}")
    print("============================\n\n")
    print("TOKEN TO ID:")
    tokens = bpe.encode(text)
    for token in tokens:
        print(f"{token}: {bpe.id2token[token]}")
    print("============================\n\n")
