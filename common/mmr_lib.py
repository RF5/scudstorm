'''
Library for finding matchmaking rating (MMR) 
and various utility functions surrounding MMR

Author: Matthew Baas
'''

import numpy as np

class MMRBracket(object):
    """
    Matchmaking bracket
    """

    def __init__(self, base_rating=1000, k_factor=40):
        """
        Runs at initialization of class object.
        @param base_rating - The rating a new player would have
        @param k_factor - The k-factor of the ELO bracket
        """
        self.base_rating = base_rating
        self.players = []
        self.k_factor = k_factor

    def __getPlayerList(self):
        """
        Returns this implementation's player list.
        @return - the list of all player objects in the implementation.
        """
        return self.players

    def getPlayer(self, name):
        """
        Returns the player in the implementation with the given name.
        @param name - name of the player to return.
        @return - the player with the given name.
        """
        for player in self.players:
            if player.name == name:
                return player
        return None

    def contains(self, name):
        """
        Returns true if this object contains a player with the given name.
        Otherwise returns false.
        @param name - name to check for.
        """
        for player in self.players:
            if player.name == name:
                return True
        return False

    def addPlayer(self, name, rating=None):
        """
        Adds a new player to the implementation.
        @param name - The name to identify a specific player.
        @param rating - The player's rating.
        """
        if rating == None:
            rating = self.base_rating

        self.players.append(_Player(name=name,rating=rating))

    def removePlayer(self, name):
        """
        Adds a new player to the implementation.
        @param name - The name to identify a specific player.
        """
        self.__getPlayerList().remove(self.getPlayer(name))


    def recordMatch(self, name1, name2, winner=None, draw=False):
        """
        Should be called after a game is played.
        @param name1 - name of the first player.
        @param name2 - name of the second player.
        """
        player1 = self.getPlayer(name1)
        player2 = self.getPlayer(name2)
        player1.games_played += 1
        player2.games_played += 1

        expected1 = player1.compareRating(player2)
        expected2 = player2.compareRating(player1)
    
        rating1 = player1.rating
        rating2 = player2.rating

        if draw:
            score1 = 0.5
            score2 = 0.5
        elif winner == name1:
            score1 = 1.0
            score2 = 0.0
        elif winner == name2:
            score1 = 0.0
            score2 = 1.0
        else:
            raise ValueError('One of the names must be the winner or draw must be True')

        newRating1 = rating1 + self.k_factor * (score1 - expected1)
        newRating2 = rating2 + self.k_factor * (score2 - expected2)

        if newRating1 < 0:
            newRating1 = 0
            newRating2 = rating2 - rating1

        elif newRating2 < 0:
            newRating2 = 0
            newRating1 = rating1 - rating2

        player1.rating = newRating1
        player2.rating = newRating2

    def getPlayerRating(self, name):
        """
        Returns the rating of the player with the given name.
        @param name - name of the player.
        @return - the rating of the player with the given name.
        """
        player = self.getPlayer(name)
        return player.rating

    def getRatingList(self):
        """
        Returns a list of tuples in the form of ({name},{rating})
        @return - the list of tuples
        """
        lst = []
        for player in self.__getPlayerList():
            lst.append((player.name, player.games_played ,player.rating))
        return lst

class _Player:
    """
    A class to represent a player in the MMR bracket
    """

    def __init__(self, name, rating):
        """
        Runs at initialization of class object.
        @param name - name of the player
        @param rating - players current mmr rating
        """
        self.name = name
        self.rating = rating
        self.games_played = 0

    def compareRating(self, opponent):
        """
        Compares the two ratings of the this player and the opponent.
        @param opponent - the player to compare against.
        @returns - The expected score between the two players.
        """
        return ( 1+10**( ( opponent.rating-self.rating )/400.0 ) ) ** -1

if __name__ == '__main__':
    print("Running MMR tests...")
    i = MMRBracket()

    i.addPlayer("Hank")
    i.addPlayer("Bill",rating=900)
    print(i.getPlayerRating("Hank"), i.getPlayerRating("Bill"))
    i.recordMatch("Hank","Bill",winner="Hank")
    print(i.getRatingList())
    i.recordMatch("Hank","Bill",winner="Bill")
    print(i.getRatingList())
    i.recordMatch("Hank","Bill",draw=True)
    print(i.getRatingList())
    i.removePlayer("Hank")
    print(i.getRatingList())

    print(">>> Basic tests completed successfully.")

    from string import ascii_lowercase
    import random
    names = list(ascii_lowercase)

    kappa = MMRBracket()
    for name in names:
        kappa.addPlayer(name)

    for j in range(1000):
        ind1, ind2 = random.sample(names, 2)
        win = ind1
        if random.random() > 0.5:
            win = ind2
        kappa.recordMatch(ind1, ind2, winner=win)

    for uu in range(10):
        ind1 = names[2]
        ind2 = random.choice(names)

        kappa.recordMatch(ind1, ind2, winner=ind1)
    
    lis = kappa.getRatingList()
    lis = sorted(lis, key = lambda elm : elm[-1], reverse=True)
    for elm in lis:
        print(elm)

    print(">>> Advanced tests completed successfully.")
