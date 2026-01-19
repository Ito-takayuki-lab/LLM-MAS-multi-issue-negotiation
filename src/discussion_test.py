__author__ = "Dong Yihan"

import os
import json

from preference_agent import preference_agent
from experiment_results import (extract_step_by_step_utilities, extract_step_by_step_attitudes_to_utilities,
                                convert_language_to_utility)
from reset_discussion_progress import reset_discussion_process_utilities, reset_discussion_analysis

agent1_analysis_dir = "./analysis/agent1.json"
agent2_analysis_dir = "./analysis/agent2.json"
agent3_analysis_dir = "./analysis/agent3.json"
agent4_analysis_dir = "./analysis/agent4.json"

agent1_combination_dir = "./combination/agent1.json"
agent2_combination_dir = "./combination/agent2.json"
agent3_combination_dir = "./combination/agent3.json"
agent4_combination_dir = "./combination/agent4.json"


def say():
    msg = "Finish"
    voice = "Victoria"
    os.system(f"say -v {voice} {msg}")
    return


if __name__ == "__main__":
    agent1 = preference_agent.PreferenceAgent(beta=1)
    agent2 = preference_agent.PreferenceAgent(beta=1)
    agent3 = preference_agent.PreferenceAgent(beta=1)
    # the second group only needs 3 agents
    # agent4 = preference_agent.PreferenceAgent(beta=1)
    for beta in range(1, 2):
        agent1.modify_beta(new_beta=beta)
        agent2.modify_beta(new_beta=beta)
        agent3.modify_beta(new_beta=beta)
        # the second group only needs 3 agents
        # agent4.modify_beta(new_beta=beta)
        print("1's beta:", agent1.beta)
        print("2's beta:", agent2.beta)
        print("3's beta:", agent3.beta)
        # the second group only needs 3 agents
        # print("4's beta:", agent4.beta)
        #  discussion experiment 10 rounds
        for i in range(1, 11):
            print(f"round {i}")
            #  numeric utilities without significance
            # agent1_post = agent1.join_discussion_reference_without_significance_weights(
            #     combination_dir=agent1_combination_dir)

            #  numeric utilities with significance weights
            # agent1_post = agent1.join_discussion_reference_with_significance_weights(
            #     combination_dir=agent1_combination_dir)

            #  linguistic direct instructions without significance
            # agent1_post = agent1.join_discussion_linguistic_without_significance(
            #     combination_dir=agent1_combination_dir)

            # linguistic description with significance
            agent1_post = agent1.join_discussion_linguistic_with_weights(combination_dir=agent1_combination_dir)

            # utility analysis without significance weights
            # analysis_1 = agent1.analyse_post_without_significance(
            #     post=agent1_post, combination_dir=agent1_combination_dir)

            #  utility analysis with significance weights
            # analysis_1 = agent1.analyse_post_with_significance_weights(
            #     post=agent1_post, combination_dir=agent1_combination_dir)

            # attitude description estimation
            # analysis_1 = agent1.analyse_post_linguistic_without_significance(
            #     post=agent1_post, combination_dir=agent1_combination_dir)

            # attitude description with significance description
            analysis_1 = agent1.analyse_post_linguistic_with_significance(
                post=agent1_post, combination_dir=agent1_combination_dir)

            preference_agent.save_analysis(analysis_dir=agent2_analysis_dir, analysis=analysis_1, agent_name="agent 1")
            preference_agent.save_analysis(analysis_dir=agent3_analysis_dir, analysis=analysis_1, agent_name="agent 1")
            preference_agent.save_analysis(analysis_dir=agent4_analysis_dir, analysis=analysis_1, agent_name="agent 1")

            # agent 2 turn
            # numeric utilities without significance
            # agent2_post = agent2.join_discussion_reference_without_significance_weights(
            #     combination_dir=agent2_combination_dir)

            # numeric utilities with significance weights
            # agent2_post = agent2.join_discussion_reference_with_significance_weights(
            #     combination_dir=agent2_combination_dir)

            # linguistic direct instructions without significance
            # agent2_post = agent2.join_discussion_linguistic_without_significance(
            #     combination_dir=agent2_combination_dir)

            # linguistic descriptions with significance weights
            agent2_post = agent2.join_discussion_linguistic_with_weights(combination_dir=agent2_combination_dir)

            #  utility analysis
            # analysis_2 = agent2.analyse_post_without_significance(
            #     post=agent2_post, combination_dir=agent2_combination_dir)

            # utility analysis with significance weights
            # analysis_2 = agent2.analyse_post_with_significance_weights(
            #     post=agent2_post, combination_dir=agent2_combination_dir)

            #  attitude description estimation
            # analysis_2 = agent2.analyse_post_linguistic_without_significance(
            #     post=agent2_post, combination_dir=agent2_combination_dir)

            # attitude description estimation with significance description
            analysis_2 = agent2.analyse_post_linguistic_with_significance(
                post=agent2_post, combination_dir=agent2_combination_dir)

            preference_agent.save_analysis(analysis_dir=agent1_analysis_dir, analysis=analysis_2, agent_name="agent 2")
            preference_agent.save_analysis(analysis_dir=agent3_analysis_dir, analysis=analysis_2, agent_name="agent 2")
            preference_agent.save_analysis(analysis_dir=agent4_analysis_dir, analysis=analysis_2, agent_name="agent 2")

            #  agent 3 turn
            #  numeric references speech without significance
            # agent3_post = agent3.join_discussion_reference_without_significance_weights(
            #     combination_dir=agent3_combination_dir)

            # numeric utilities with significance weights
            # agent3_post = agent3.join_discussion_reference_with_significance_weights(
            #     combination_dir=agent3_combination_dir)

            #  linguistic direct instructions without significance
            # agent3_post = agent3.join_discussion_linguistic_without_significance(
            #     combination_dir=agent3_combination_dir)

            # linguistic descriptions with significance weights
            agent3_post = agent3.join_discussion_linguistic_with_weights(combination_dir=agent3_combination_dir)

            #  utility analysis without significance weights
            # analysis_3 = agent3.analyse_post_without_significance(
            #     post=agent3_post, combination_dir=agent3_combination_dir)

            # utility analysis with significance weights
            # analysis_3 = agent3.analyse_post_with_significance_weights(
            #     post=agent3_post, combination_dir=agent3_combination_dir)

            #  attitude description estimation
            # analysis_3 = agent3.analyse_post_linguistic_without_significance(
            #     post=agent3_post, combination_dir=agent3_combination_dir)

            # attitude description estimation with significance weights
            analysis_3 = agent3.analyse_post_linguistic_with_significance(
                post=agent3_post, combination_dir=agent3_combination_dir)

            preference_agent.save_analysis(analysis_dir=agent1_analysis_dir, analysis=analysis_3, agent_name="agent 3")
            preference_agent.save_analysis(analysis_dir=agent2_analysis_dir, analysis=analysis_3, agent_name="agent 3")
            preference_agent.save_analysis(analysis_dir=agent4_analysis_dir, analysis=analysis_3, agent_name="agent 3")

            #  agent 4 turn
            #  numeric utilities without significance
            # agent4_post = agent4.join_discussion_reference_without_significance_weights(
            #     combination_dir=agent4_combination_dir)

            #  linguistic direct instructions without significance
            # agent4_post = agent4.join_discussion_linguistic_without_significance(
            #     combination_dir=agent4_combination_dir)

            #  utility analysis
            # analysis_4 = agent4.analyse_post_without_significance(
            #     post=agent4_post, combination_dir=agent4_combination_dir)

            #  attitude description estimation
            # analysis_4 = agent4.analyse_post_linguistic_without_significance(
            #     post=agent4_post, combination_dir=agent4_combination_dir)

            # preference_agent.save_analysis(analysis_dir=agent1_analysis_dir, analysis=analysis_4, agent_name="agent 4")
            # preference_agent.save_analysis(analysis_dir=agent2_analysis_dir, analysis=analysis_4, agent_name="agent 4")
            # preference_agent.save_analysis(analysis_dir=agent3_analysis_dir, analysis=analysis_4, agent_name="agent 4")

            print("\npost 1\n", agent1_post)
            print("\npost 2\n", agent2_post)
            print("\npost 3\n", agent3_post)

            #   メカニズム付け言語モデルに基づくエージェント
            # agent1.adjust_utilities_with_simple_limitation(analysis_dir=agent1_analysis_dir,
            #                                                combination_dir=agent1_combination_dir)
            # agent2.adjust_utilities_with_simple_limitation(analysis_dir=agent2_analysis_dir,
            #                                                combination_dir=agent2_combination_dir)
            # agent3.adjust_utilities_with_simple_limitation(analysis_dir=agent3_analysis_dir,
            #                                                combination_dir=agent3_combination_dir)
            # agent4.adjust_utilities_with_simple_limitation(analysis_dir=agent4_analysis_dir,
            #                                                combination_dir=agent4_combination_dir)

            #  効用値付け言語モデルに基づくエージェント（メカニズムなし）
            # agent1.adjust_utilities_llm_with_reference(analysis_dir=agent1_analysis_dir,
            #                                            combination_dir=agent1_combination_dir)
            # agent2.adjust_utilities_llm_with_reference(analysis_dir=agent2_analysis_dir,
            #                                            combination_dir=agent2_combination_dir)
            # agent3.adjust_utilities_llm_with_reference(analysis_dir=agent3_analysis_dir,
            #                                            combination_dir=agent3_combination_dir)
            # agent4.adjust_utilities_llm_with_reference(analysis_dir=agent4_analysis_dir,
            #                                            combination_dir=agent4_combination_dir)

            #  言語モデルに基づくエージェント（baseline）
            # agent1.adjust_attitudes_llm_with_description(analysis_dir=agent1_analysis_dir,
            #                                              combination_dir=agent1_combination_dir)
            # agent2.adjust_attitudes_llm_with_description(analysis_dir=agent2_analysis_dir,
            #                                              combination_dir=agent2_combination_dir)
            # agent3.adjust_attitudes_llm_with_description(analysis_dir=agent3_analysis_dir,
            #                                              combination_dir=agent3_combination_dir)
            # agent4.adjust_attitudes_llm_with_description(analysis_dir=agent4_analysis_dir,
            #                                              combination_dir=agent4_combination_dir)

            #  譲歩戦略を加え、メカニズム付け言語モデルに基づくエージェント
            # agent1.adjust_utilities_with_concession_strategy(
            #     analysis_dir=agent1_analysis_dir, combination_dir=agent1_combination_dir)
            # agent2.adjust_utilities_with_concession_strategy(
            #     analysis_dir=agent2_analysis_dir, combination_dir=agent2_combination_dir)
            # agent3.adjust_utilities_with_concession_strategy(
            #     analysis_dir=agent3_analysis_dir, combination_dir=agent3_combination_dir)

            #  譲歩戦略を加え、効用値付け言語モデルに基づくエージェント
            # agent1.adjust_utilities_llm_with_reference_and_concession(
            #     analysis_dir=agent1_analysis_dir, combination_dir=agent1_combination_dir)
            # agent2.adjust_utilities_llm_with_reference_and_concession(
            #     analysis_dir=agent2_analysis_dir, combination_dir=agent2_combination_dir)
            # agent3.adjust_utilities_llm_with_reference_and_concession(
            #     analysis_dir=agent3_analysis_dir, combination_dir=agent3_combination_dir)

            #  譲歩戦略を加え、言語モデルに基づくエージェント(baseline)
            agent1.adjust_attitudes_llm_with_description_and_concession(
                analysis_dir=agent1_analysis_dir, combination_dir=agent1_combination_dir)
            agent2.adjust_attitudes_llm_with_description_and_concession(
                analysis_dir=agent2_analysis_dir, combination_dir=agent2_combination_dir)
            agent3.adjust_attitudes_llm_with_description_and_concession(
                analysis_dir=agent3_analysis_dir, combination_dir=agent3_combination_dir)

        agent1_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent1_combination_dir)
        agent2_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent2_combination_dir)
        agent3_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent3_combination_dir)
        # agent4_utility_history = extract_step_by_step_utilities(combination_dir=agent4_combination_dir)
        experiments_results_dir = f"experiments_results/with_concession_2/group_3_4.1/beta{beta}.json"
        print(experiments_results_dir)

        experiments_results = {
            "agent_1_utility": agent1_utility_history,
            "agent_2_utility": agent2_utility_history,
            "agent_3_utility": agent3_utility_history
            # No agent 4 with concession
            # "agent_4_utility": agent4_utility_history
        }

        with open(experiments_results_dir, "w") as f:
            json.dump(experiments_results, f, indent=4)

        reset_discussion_process_utilities(combination_dir=agent1_combination_dir)
        reset_discussion_process_utilities(combination_dir=agent2_combination_dir)
        reset_discussion_process_utilities(combination_dir=agent3_combination_dir)
        reset_discussion_process_utilities(combination_dir=agent4_combination_dir)

        reset_discussion_analysis(analysis_dir=agent1_analysis_dir)
        reset_discussion_analysis(analysis_dir=agent2_analysis_dir)
        reset_discussion_analysis(analysis_dir=agent3_analysis_dir)
        reset_discussion_analysis(analysis_dir=agent4_analysis_dir)
    say()
    # else:
    #     agent1.judge_consensus_type(combination_dir=agent1_combination_dir)
    #     agent2.judge_consensus_type(combination_dir=agent2_combination_dir)
    #     agent3.judge_consensus_type(combination_dir=agent3_combination_dir)
    #     agent4.judge_consensus_type(combination_dir=agent4_combination_dir)
