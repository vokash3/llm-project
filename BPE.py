class BPE:
    """
    Класс для реализации алгоритма Byte Pair Encoding (BPE).

    Атрибуты:
        vocab_size (int): Желаемый размер словаря токенов.
        id2token (dict): Словарь для преобразования идентификатора токена в сам токен.
        token2id (dict): Словарь для преобразования токена в его идентификатор.
    """
    def __init__(self, vocab_size: int):
        """
        Инициализирует экземпляр BPE с заданным размером словаря.

        Args:
            vocab_size (int): Желаемый размер словаря токенов.
        """
        self.vocab_size = vocab_size

    def fit(self, text: str):
        """
        Обучает модель BPE на заданном тексте, создавая словарь токенов.

        Алгоритм итеративно объединяет наиболее частые пары символов (или уже созданных токенов)
        до тех пор, пока размер словаря не достигнет vocab_size или пока есть пары для объединения.

        Args:
            text (str): Текст, на котором будет обучаться модель.

        Примечания:
            После вызова этого метода инициализируются атрибуты id2token и token2id.
        """
        tokens = sorted(set(text))
        sequence = list(text)

        while len(tokens) < self.vocab_size:
            pair_counts = {}

            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]
            tokens.append(new_token)

            new_sequence = []
            i = 0

            while i < len(sequence):
                if (
                    i < len(sequence) - 1
                    and sequence[i] == best_pair[0]
                    and sequence[i + 1] == best_pair[1]
                ):
                    new_sequence.append(new_token)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1

            sequence = new_sequence

        self.id2token = {
            token_id: token
            for token_id, token in enumerate(tokens)
        }
        """dict: Словарь, сопоставляющий идентификатор токена (int) с самим токеном (str)."""

        self.token2id = {
            token: token_id
            for token_id, token in self.id2token.items()
        }
        """dict: Словарь, сопоставляющий токен (str) с его идентификатором (int)."""