class Cell:
    def __init__(self):
        self.state = 0

    def state_change(self, state: int):
        self.state = state

    def state_rnd_by_p(self, p: float):
        """ Method selecting state of the cell with probability p """