from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef


class Tokenization:
    def __init__(self, token_def: PostfixNotationTokenDef = PostfixNotationTokenDef()) -> None:
        self._token_def = token_def

    @property
    def token_def(self) -> PostfixNotationTokenDef:
        return self._token_def
