def main() -> None:
    #make sure that the module is located somewhere where your Python system looks for packages
    #note that python does not search directory trees, hence you must provide the mother-directory of the package

    #importing the module
    from .hex_engine import hexPosition

    #initializing a game object
    game = hexPosition()

    #this is how your agent can be imported
    #'submission' is the (sub)package that you provide
    #please use a better name that identifies your group
    from .submission.facade import agent4, agent3, agent2, agent1

    #make sure that the agent you have provided is such that the following three
    #method-calls are error-free and as expected

    from .submission.rule_based_agent_4 import announce_agent_color
    announce_agent_color(game.board)

    #let your agent play against random
    #game.human_vs_machine(human_player=1, machine=agent)
    game.machine_vs_machine(machine1=agent4, machine2=agent4)

    # experimental! 
    from .submission.rule_based_agent_4 import print_strategy_summary
    print_strategy_summary()




    #from random import choice
    #game.machine_vs_machine(machine1=lambda board, moves: choice(moves), machine2=None)

    #let your agent play against itself
    #game.machine_vs_machine(machine1=agent, machine2=agent)
