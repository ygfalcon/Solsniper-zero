class Instruction:
    def __init__(self, program_id, data: bytes, keys: list):
        self.program_id = program_id
        self.data = data
        self.keys = keys
