from .rule_based_agent_1 import rule_based_agent_1
from .rule_based_agent_2 import rule_based_agent_2
from .rule_based_agent_3 import rule_based_agent_3
from .rule_based_agent_4 import rule_based_agent_4

#rule-based agent growing
def agent4 (board, action_set):
    return rule_based_agent_4(board, action_set)

def agent3 (board, action_set):
    return rule_based_agent_3(board, action_set)

def agent2 (board, action_set):
    return rule_based_agent_2(board, action_set)

def agent1 (board, action_set):
    return rule_based_agent_1(board, action_set)

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.