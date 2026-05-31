from BPE import BPE

vocab_size: int = 20
text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
tokenizer_path: str = './data/bpe.dill'
bpe: BPE

if __name__ == '__main__':
    try:
        bpe = BPE.load('data/bpe.dill')
        print(f"✅ Файл токенайзера {tokenizer_path} ЗАГРУЖЕН!")
    except FileNotFoundError:
        print(f"⛔ Файл токенайзера {tokenizer_path} НЕ ОБНАРУЖЕН!")
        print(f"ℹ️ Создаём новый токенайзер...")
        bpe = BPE(vocab_size=vocab_size)
        bpe.fit(text)  # делаем токены из текста
        print("ℹ️ Сохраняем токенайзер: ")
        bpe.save(tokenizer_path)
        print(f"✅ Файл токенайзера {tokenizer_path} сохранён!")
    print("TOKEN TO ID:")
    for id, token in enumerate(bpe.token2id):
        print(f"{id}: {token}")
    print("============================\n\n")
    print("TOKEN TO ID:")
    tokens = bpe.encode(text)
    for token in tokens:
        print(f"{token}: {bpe.id2token[token]}")
    print("============================\n\n")
    print("TOKENS TO WORDS (DECODE):")
    print(bpe.decode(tokens))
    print("============================\n\n")
