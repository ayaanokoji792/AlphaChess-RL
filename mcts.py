import math

class Node:
    def __init__(self, prior_prob):
        """
        prior_prob: The initial probability of this move, predicted by the neural network's Policy Head.
        """
        self.visit_count = 0       # How many times MCTS simulated this path
        self.value_sum = 0         # The total value accumulated from this path
        self.prior_prob = prior_prob 
        self.children = {}         # Dictionary of legal moves from this position
        
    def get_value(self):
        """Returns the average win/loss value of this position."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, parent_visit_count):
        """
        Upper Confidence Bound (UCB) formula.
        This balances EXPLORATION (trying new moves) vs EXPLOITATION (playing known good moves).
        """
        c_puct = 1.25 # Exploration constant
        
        exploitation = self.get_value() #calculates the effectiveness of this move based on past simulations
        
        exploration = c_puct * self.prior_prob * (math.sqrt(parent_visit_count) / (1 + self.visit_count)) #checks whether this move has been explored enough compared to its prior probability
        
        return exploitation + exploration