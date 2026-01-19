__author__ = "Dong Yihan"

import json

from preference_agent import preference_agent

agent1_combination_dir = "./combination/agent1.json"
agent2_combination_dir = "./combination/agent2.json"
agent3_combination_dir = "./combination/agent3.json"
agent4_combination_dir = "./combination/agent4.json"

if __name__ == "__main__":
    agent1 = preference_agent.PreferenceAgent(beta=1)
    agent2 = preference_agent.PreferenceAgent(beta=1)
    agent3 = preference_agent.PreferenceAgent(beta=1)

    # agent1_post_with_references = agent1.join_discussion_reference_with_significance_weights(
    #     combination_dir=agent1_combination_dir)
    # print("test post 1 with references:\n", agent1_post_with_references)
    #
    # agent1_analysis_with_references = agent1.analyse_post_with_significance_weights(
    #     post=agent1_post_with_references, combination_dir=agent1_combination_dir)
    #
    # agent1_analysis_with_references = agent1_analysis_with_references.strip()
    # print(agent1_analysis_with_references.startswith("```json"))
    # if agent1_analysis_with_references.startswith("```json"):
    #     agent1_analysis_with_references = agent1_analysis_with_references.replace("```json", "")
    #     agent1_analysis_with_references = agent1_analysis_with_references.replace("```", "")
    # print("\n\ntest analysis 1 with references:\n", agent1_analysis_with_references)
    # agent1_analysis_list = json.loads(agent1_analysis_with_references)
    # print("\n\ntest analysis converted list:\n", agent1_analysis_list)

    # agent2_post_with_references = agent2.join_discussion_reference_with_significance_weights(
    #     combination_dir=agent2_combination_dir)
    # print("\n\ntest post 2 with references:\n", agent2_post_with_references)
    #
    # agent2_analysis_with_references = agent2.analyse_post_with_significance_weights(
    #     post=agent2_post_with_references, combination_dir=agent2_combination_dir)
    # print("\n\ntest analysis 2 with references:\n", agent2_analysis_with_references)
    #
    # agent3_post_with_references = agent3.join_discussion_reference_with_significance_weights(
    #     combination_dir=agent3_combination_dir)
    # print("\n\ntest post 3 with references:\n", agent3_post_with_references)
    #
    # agent3_analysis_with_references = agent3.analyse_post_with_significance_weights(
    #     post=agent3_post_with_references, combination_dir=agent3_combination_dir)
    # print("\n\ntest analysis 3 with references:\n", agent3_analysis_with_references)

    # agent1_post_with_descriptions = agent1.join_discussion_linguistic_with_weights(
    #     combination_dir=agent1_combination_dir)
    #
    # print("\n\ntest post with descriptions:\n", agent1_post_with_descriptions)
    #
    # agent1_analysis_with_descriptions = agent1.analyse_post_linguistic_with_significance(
    #     post=agent1_post_with_descriptions, combination_dir=agent1_combination_dir)
    #
    # print("\n\ntest analysis with descriptions:\n", agent1_analysis_with_descriptions)
