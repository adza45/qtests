class Unit():
    def __init__(self):
        pass

    def set_location(self, location):
        self.location = location

class Player(Unit):
    def __init__(self, map_size):
        self.color = (255, 255, 255)
        self.size = (5, 5)
        self.map_size = map_size
        self.score = 0
        self.inital_location = (0, 0)
        self.restart = False
        Unit.__init__(self)

    def move_right(self):
        if self.location[0] < self.map_size[0] - self.size[0]:
            self.location = (self.location[0] + 5, self.location[1])

    def move_down(self):
        if self.location[1] < self.map_size[1] - self.size[1]:
            self.location = (self.location[0], self.location[1] + 5)

    def move_left(self):
        if self.location[0] > 0:
            self.location = (self.location[0] - 5, self.location[1])

    def move_up(self):
        if self.location[1] > 0:
            self.location = (self.location[0], self.location[1] - 5)
class Objective(Unit):
    def __init__(self):
        self.color = (128, 128, 128)
        Unit.__init__(self)

class Mine(Unit):
    def __init__(self):
        self.color = (64, 64, 64)
        Unit.__init__(self)
