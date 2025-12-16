class TextBuffer:
    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def backspace(self):
        if self.tokens:
            self.tokens.pop()

    def clear(self):
        self.tokens = []

    def get_sentence(self):
        if not self.tokens:
            return ""
        return " ".join(self.tokens).capitalize() + "."
