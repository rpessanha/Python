class player:
    def __init__(self, name, power):
        self.name = name
        self.power = power


name = input('What is the name of the player:')
power = input('What is the power of the player:')
player1 = player(name, power)
print(player1.name)