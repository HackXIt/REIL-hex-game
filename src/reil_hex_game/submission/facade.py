from .rule_based_agent import rule_based_agent

#trivial solution
def agent (board, action_set):
    return rule_based_agent(board, action_set)

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.