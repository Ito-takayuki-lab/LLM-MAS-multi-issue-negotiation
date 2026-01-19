__author__ = "Dong Yihan"

import json


def reset_discussion_process_utilities(combination_dir):
    agent_utility_history_dir = combination_dir
    with open(agent_utility_history_dir, "r") as f1:
        agent_utility_history = json.load(f1)

    initial_agent_utilities = agent_utility_history[:3]  # number of opinions

    with open(agent_utility_history_dir, "w") as f2:
        json.dump(initial_agent_utilities, f2, indent=4)

    return


def reset_discussion_analysis(analysis_dir):
    agent_analysis_history_dir = analysis_dir
    with open(agent_analysis_history_dir, "w") as f1:
        reset_data = []
        json.dump(reset_data, f1, indent=4)

    return


def reset_questionnaire_answers(answers_dir, combination_dir, comparison_dir):
    reset_data = []
    with open(answers_dir, "w") as f1:
        json.dump(reset_data, f1, indent=4)

    with open(combination_dir, "w") as f2:
        json.dump(reset_data, f2, indent=4)

    with open(comparison_dir, "w") as f3:
        json.dump(reset_data, f3, indent=4)

    return


if __name__ == "__main__":
    agent1_history_dir = "combination/agent1.json"
    agent2_history_dir = "combination/agent2.json"
    agent3_history_dir = "combination/agent3.json"
    agent4_history_dir = "combination/agent4.json"

    agent1_analysis_history_dir = "analysis/agent1.json"
    agent2_analysis_history_dir = "analysis/agent2.json"
    agent3_analysis_history_dir = "analysis/agent3.json"
    agent4_analysis_history_dir = "analysis/agent4.json"

    answers_test_dir = "./questionnaire/answers_test.json"
    combination_test_dir = "./questionnaire/combination_test.json"
    comparison_dir = "./questionnaire/comparison_test.json"

    reset_discussion_process_utilities(combination_dir=agent1_history_dir)
    reset_discussion_process_utilities(combination_dir=agent2_history_dir)
    reset_discussion_process_utilities(combination_dir=agent3_history_dir)
    reset_discussion_process_utilities(combination_dir=agent4_history_dir)

    reset_discussion_analysis(analysis_dir=agent1_analysis_history_dir)
    reset_discussion_analysis(analysis_dir=agent2_analysis_history_dir)
    reset_discussion_analysis(analysis_dir=agent3_analysis_history_dir)
    reset_discussion_analysis(analysis_dir=agent4_analysis_history_dir)

    reset_questionnaire_answers(answers_dir=answers_test_dir, combination_dir=combination_test_dir,
                                comparison_dir=comparison_dir)
