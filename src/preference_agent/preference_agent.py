__author__ = "Dong Yihan"

import json
import os
import copy
from collections import defaultdict
from .llm_setting import OPENAI


def convert_analysis(analysis_str: str):
    """
    refine it!
    modify the method to address element index to the analysis results
    :param analysis_str:
    :return:
    """
    analysis_str = analysis_str.strip()
    if "[" in analysis_str:
        if analysis_str.startswith("```"):
            analysis_str = analysis_str.replace("```", "")
        if analysis_str.startswith("json"):
            analysis_str = analysis_str[len("json"):].strip()
        analysis_list = json.loads(analysis_str)
    else:
        analysis_list = []
        analysis_str = analysis_str.splitlines()
        for substr in analysis_str:
            analysis_list.append(json.loads(substr))
    for index, element in enumerate(analysis_list):
        element["index"] = index + 1

    return analysis_list


def save_analysis(analysis_dir, analysis, agent_name):
    analysis_list = convert_analysis(analysis_str=analysis)
    data = []
    with open(analysis_dir, "r") as f:
        data = json.load(f)

    for analysis_result in analysis_list:
        analysis_result["agent name"] = agent_name
    data = data + analysis_list

    with open(analysis_dir, "w") as f:
        json.dump(data, f, indent=4)
    return


def calculate_average_utilities(analysis_dir, combination_dir, opinion_index: int, other_agents_number: int):
    """
    calculate the average attitudes of other agents on specific opinions with numeric references
    :param combination_dir:
    :param analysis_dir: directory of analysis results
    :param opinion_index: the index of the specific opinion.
    :param other_agents_number: the number of total agents - 1 to calculate environmental average utilities
    :return:
    """
    other_agents_analysis_results_list = load_analysis_results(analysis_dir=analysis_dir)
    other_agents_corresponding_list = [element["attitude"] for element in other_agents_analysis_results_list if
                                       (element["index"] == opinion_index)]
    latest_opinions_and_utilities = load_latest_opinions_attitudes_importance(
        combination_dir=combination_dir, opinion_numbers=3)
    corresponding_utility_dic = next((d for d in latest_opinions_and_utilities if d["index"] == opinion_index), None)
    corresponding_utility = corresponding_utility_dic["attitude"]

    agents_average_utility = (sum(other_agents_corresponding_list[-other_agents_number:]) + corresponding_utility) / (
            other_agents_number + 1)
    agents_average_utility = round(agents_average_utility, 2)

    # print(f"opinion {opinion_index} average:", agents_average_utility)

    return agents_average_utility


def calculate_sum_concession(analysis_dir, combination_dir, opinion_index: int, other_agents_number: int):
    """
    calculate the concession utilities sum
    multiply beta and calculate the average adjusted utilities outside the function()
    :param analysis_dir:
    :param combination_dir:
    :param opinion_index:
    :param other_agents_number:
    :return:
    """
    other_agents_analysis_results = load_analysis_results(analysis_dir=analysis_dir)
    other_agents_corresponding_list = [element for element in other_agents_analysis_results if
                                       (element["index"] == opinion_index)]
    latest_opinions_and_utilities = load_latest_opinions_attitudes_importance(
        combination_dir=combination_dir, opinion_numbers=3)
    corresponding_dic = next((d for d in latest_opinions_and_utilities if d["index"] == opinion_index), None)
    corresponding_utility = corresponding_dic["attitude"]
    corresponding_significance = corresponding_dic["importance"]
    zeta = 1.0  # parameter for calculating relative priority
    latest_other_agents_corresponding_list = other_agents_corresponding_list[-other_agents_number:]

    relative_concession_utility_general = 0
    #  should be 2 corresponding utilities belonging to the other 2 agents
    print("latest other agents analysis results", latest_other_agents_corresponding_list)
    print(opinion_index)

    for latest_other_agents_analysis in latest_other_agents_corresponding_list:
        relative_significance = latest_other_agents_analysis["significance"] / (corresponding_significance + zeta)
        relative_utility_difference = latest_other_agents_analysis["attitude"] - corresponding_utility
        relative_concession_utility = relative_significance * relative_utility_difference
        relative_concession_utility_general = relative_concession_utility_general + relative_concession_utility

    return round(relative_concession_utility_general, 2)


def combine_others_linguistic_attitudes(analysis_dir, opinion_number: int):
    """
    combine the attitudes descriptions of other agents as a 2-dimensional array to be loaded in the prompts
    :param analysis_dir:
    :param opinion_number:
    :return:
    """
    other_agents_analysis_results_list = load_analysis_results(analysis_dir=analysis_dir)
    other_agents_attitudes_2d_list = []
    for opinion_index in range(1, opinion_number + 1):
        other_agents_corresponding_list = [element["attitude"] for element in other_agents_analysis_results_list if
                                           (element["index"] == opinion_index)]
        other_agents_attitudes_2d_list.append(other_agents_corresponding_list)

    return other_agents_attitudes_2d_list


def adjust_preference_only_average_response(analysis_dir, combination_dir):
    """
    The function to adjust the utility of the agents upon different opinions.
    This function follows the every basic "average attitude response mechanism"
    :param analysis_dir: The directory where the analysis results of other agents utilities are stored.
    :param combination_dir: The directory where the utilities of agents are stored.
    :return:
    """
    with open(analysis_dir, "r") as f1:
        other_agents_list = json.load(f1)

    other_agents_q1_list = [element["attitude"] for element in other_agents_list if element["index"] == 1]
    other_agents_q2_list = [element["attitude"] for element in other_agents_list if element["index"] == 2]
    other_agents_q3_list = [element["attitude"] for element in other_agents_list if element["index"] == 3]

    # The average attitudes are from the latest recordings.
    # The number is 3 because there are 4 agents are listed in the experiments
    # This part also needs to be refined in case there are multiple (not a predefined number) opinions included.
    q1_average = sum(other_agents_q1_list[-3:]) / 3
    q2_average = sum(other_agents_q2_list[-3:]) / 3
    q3_average = sum(other_agents_q3_list[-3:]) / 3

    with open(combination_dir, "r") as f2:
        combination_history = json.load(f2)
    data = [copy.deepcopy(item) for item in combination_history[-3:]]
    data[0]["attitude"] = q1_average
    data[1]["attitude"] = q2_average
    data[2]["attitude"] = q3_average
    combination_history = combination_history + data

    with open(combination_dir, "w") as f3:
        json.dump(combination_history, f3, indent=4)

    return


def load_latest_opinions_attitudes_importance(combination_dir: str, opinion_numbers: int):
    opinions_attitudes_importance_dir = combination_dir
    with open(opinions_attitudes_importance_dir, "r") as f:
        latest_opinions_attitudes_importance_list = json.load(f)
        latest_opinions_attitudes_importance_list = latest_opinions_attitudes_importance_list[-opinion_numbers:]

    return latest_opinions_attitudes_importance_list


def load_opinions(combination_dir, opinion_numbers: int):
    opinions_attitudes_importance_dir = combination_dir
    latest_opinions_attitudes_importance_list = load_latest_opinions_attitudes_importance(
        combination_dir=opinions_attitudes_importance_dir, opinion_numbers=opinion_numbers)

    opinion_list = [e["opinion"] for e in latest_opinions_attitudes_importance_list]

    return opinion_list


def load_initial_opinions_attitudes_importance(combination_dir: str, opinion_numbers: int):
    with open(combination_dir, "r") as file:
        opinions_and_attitudes = json.load(file)
    initial_opinions_and_attitudes = opinions_and_attitudes[:opinion_numbers]

    return initial_opinions_and_attitudes


def load_analysis_results(analysis_dir):
    with open(analysis_dir, "r") as f:
        other_agents_analysis_results = json.load(f)

    return other_agents_analysis_results


def separate_analysis_results_by_opinion_index(analysis_data: list):
    """
    separate analysis results into small lists by different agent names and sort them
    :param analysis_data:
    :return: 2-d array. Each element is the combination of analysis data of an agent
    """
    agent_name_key = "index"
    grouped = defaultdict(list)
    for item in analysis_data:
        grouped[item[agent_name_key]].append(item)

    sorted_grouped_items = sorted(grouped.items(), key=lambda x: x[0])

    result_groups = [group for _, group in sorted_grouped_items]

    return result_groups


class PreferenceAgent:

    def __init__(self, beta=1, gamma=1):
        """

        :param beta:
        :param gamma:
        """
        self.status = {
            "status": 500,
            "message": "error in initialization of preference agent"
        }

        self.openai = OPENAI()

        self.beta = beta  # acceptable consensus parameter

        # The tolerance consensus function is a quadratic function (二次関数), where gamma is the top point value when
        # the utility (x) = 0, which means that how tolerant this person is to other attitudes when they are neutral.
        # When the utility (x) = -2/2 (extremely disagree or agree), then the tolerance value is 0
        self.gamma = gamma  # tolerance consensus parameter

        post_instruction_dir = "./prompt/post_generation_without_weights.json"
        with open(post_instruction_dir, "r") as f11:
            post_instruction_data = json.load(f11)
        self.post_instruction = {
            "system": post_instruction_data["system"],
            "user": post_instruction_data["user"]
        }

        post_instruction_with_weights_dir = "./prompt/post_generation_with_weights.json"
        with open(post_instruction_with_weights_dir, "r") as f12:
            post_instruction_with_weights_data = json.load(f12)
        self.post_with_weights_instruction = {
            "system": post_instruction_with_weights_data["system"],
            "user": post_instruction_with_weights_data["user"]
        }

        post_instruction_linguistic_dir = "./prompt/post_generation_linguistic_without_weights.json"
        with open(post_instruction_linguistic_dir, "r") as f13:
            post_instruction_linguistic_data = json.load(f13)
        self.post_linguistic_instruction = {
            "system": post_instruction_linguistic_data["system"],
            "user": post_instruction_linguistic_data["user"]
        }

        post_instruction_linguistic_with_weights_dir = "./prompt/post_generation_linguistic_with_weights.json"
        with open(post_instruction_linguistic_with_weights_dir, "r") as f14:
            post_instruction_linguistic_with_weights_data = json.load(f14)
        self.post_linguistic_with_weights_instruction = {
            "system": post_instruction_linguistic_with_weights_data["system"],
            "user": post_instruction_linguistic_with_weights_data["user"]
        }

        analysis_instruction_dir = "./prompt/post_analysis_without_weights.json"
        with open(analysis_instruction_dir, "r") as f21:
            analysis_data = json.load(f21)
        self.analysis_instruction = {
            "system": analysis_data["system"],
            "user": analysis_data["user"]
        }

        # without markup
        # analysis_instruction_with_weights_dir = "./prompt/post_analysis_with_weights.json"
        # with markups
        analysis_instruction_with_weights_dir = "./prompt/post_analysis_with_weights_markup.json"
        with open(analysis_instruction_with_weights_dir, "r") as f22:
            analysis_with_weights_data = json.load(f22)
        self.analysis_with_weights_instruction = {
            "system": analysis_with_weights_data["system"],
            "user": analysis_with_weights_data["user"]
        }

        analysis_instruction_linguistic_dir = "./prompt/post_analysis_linguistic_without_weights.json"
        with open(analysis_instruction_linguistic_dir, "r") as f23:
            analysis_linguistic_data = json.load(f23)
        self.analysis_linguistic_instruction = {
            "system": analysis_linguistic_data["system"],
            "user": analysis_linguistic_data["user"]
        }

        analysis_instruction_linguistic_with_weights_dir = "./prompt/post_analysis_linguistic_with_weights.json"
        with open(analysis_instruction_linguistic_with_weights_dir, "r") as f24:
            analysis_linguistic_with_weights_data = json.load(f24)
        self.analysis_linguistic_with_weights_instruction = {
            "system": analysis_linguistic_with_weights_data["system"],
            "user": analysis_linguistic_with_weights_data["user"]
        }

        utility_adjustment_with_references_instruction_dir = "./prompt/utility_adjustment_with_references.json"
        with open(utility_adjustment_with_references_instruction_dir, "r") as f31:
            utility_adjustment_with_references_instruction_data = json.load(f31)
        self.utility_adjustment_with_references_instruction = {
            "system": utility_adjustment_with_references_instruction_data["system"],
            "user": utility_adjustment_with_references_instruction_data["user"]
        }

        utility_adjustment_with_references_and_concession_instruction_dir = "./prompt/utility_adjustment_with_references_and_concession.json"
        with open(utility_adjustment_with_references_and_concession_instruction_dir, "r") as f32:
            utility_adjustment_with_references_and_concession_instruction_data = json.load(f32)
        self.utility_adjustment_with_references_and_concession_instruction = {
            "system": utility_adjustment_with_references_and_concession_instruction_data["system"],
            "user": utility_adjustment_with_references_and_concession_instruction_data["user"]
        }

        attitude_adjustment_with_description_instruction_dir = "./prompt/preference_adjustment_with_descriptions.json"
        with open(attitude_adjustment_with_description_instruction_dir, "r") as f41:
            attitude_adjustment_with_description_instruction_data = json.load(f41)
        self.attitude_adjustment_with_description_instruction = {
            "system": attitude_adjustment_with_description_instruction_data["system"],
            "user": attitude_adjustment_with_description_instruction_data["user"]
        }

        attitude_adjustment_with_description_and_concession_instruction_dir = "./prompt/preference_adjustment_with_descriptions_and_concession.json"
        with open(attitude_adjustment_with_description_and_concession_instruction_dir, "r") as f42:
            attitude_adjustment_with_description_and_concession_instruction_data = json.load(f42)
        self.attitude_adjustment_with_description_and_concession_instruction = {
            "system": attitude_adjustment_with_description_and_concession_instruction_data["system"],
            "user": attitude_adjustment_with_description_and_concession_instruction_data["user"]
        }

        #  bases template for linguistic analysis
        self.bases_list_analysis = [
            "Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"
        ]

        #  bases template for linguistic attitude adjustment
        self.bases_list_adjustment = [
            "Strongly Disagree", "Disagree", "Slightly Disagree", "Neutral", "Slightly Agree", "Agree", "Strongly Agree"
        ]

        #  modifiers template for linguistic analysis
        self.modifiers_list = [
            "Definitely", "Probably", "Somewhat between...", "Slightly leaning toward"
        ]

        # significance description terms
        # self.significance_terms = [
        #     "Extremely important", "Very important", "Moderately important", "Slightly important", "Not important at all"
        # ]

        self.status.update(status=200, message="successfully initialize the preference agent.")
        return

    def modify_beta(self, new_beta):
        self.beta = new_beta
        return

    def beta_to_acceptability_descriptions(self):

        if self.beta >= 18:
            return "Extremely open to attitude adjustment"
        elif self.beta >= 14:
            return "Very open to attitude adjustment"
        elif self.beta >= 10:
            return "Moderately open to attitude adjustment"
        elif self.beta >= 6:
            return "Somewhat cautious about attitude adjustment"
        elif self.beta >= 2:
            return "Reluctant to adjust attitude"
        else:
            return "Very reluctant to adjust attitude"

    def join_discussion_reference_without_significance_weights(self, combination_dir):
        self.status.update(status=500, message="error in join_discussion_test")

        opinions_attitudes_importance_list = load_latest_opinions_attitudes_importance(combination_dir=combination_dir,
                                                                                       opinion_numbers=3)
        opinions_list = [element["opinion"] for element in opinions_attitudes_importance_list]
        attitudes_list = [element["attitude"] for element in opinions_attitudes_importance_list]

        post_prompts = [
            {
                "role": "system",
                "content": self.post_instruction["system"]
            },
            {
                "role": "user",
                "content": self.post_instruction["user"].format(
                    opinions_list=opinions_list, attitudes_list=attitudes_list
                )
            }
        ]

        discussion_contents = self.openai.get_gpt_response(prompts=post_prompts)

        self.status.update(status=200, message="successfully generate responses")
        return discussion_contents

    def join_discussion_reference_with_significance_weights(self, combination_dir):

        opinions_attitudes_importance_list = load_latest_opinions_attitudes_importance(combination_dir=combination_dir,
                                                                                       opinion_numbers=3)
        opinions_list = [element["opinion"] for element in opinions_attitudes_importance_list]
        attitudes_list = [element["attitude"] for element in opinions_attitudes_importance_list]
        significance_weights_list = [element["importance"] for element in opinions_attitudes_importance_list]

        post_prompts = [
            {
                "role": "system",
                "content": self.post_with_weights_instruction["system"]
            },
            {
                "role": "user",
                "content": self.post_with_weights_instruction["user"].format(
                    opinions_list=opinions_list, attitudes_list=attitudes_list,
                    significance_weights_list=significance_weights_list
                )
            }
        ]

        discussion_contents = self.openai.get_gpt_response(prompts=post_prompts)

        return discussion_contents

    def join_discussion_linguistic_without_significance(self, combination_dir):

        opinions_attitudes_importance_list = load_latest_opinions_attitudes_importance(combination_dir=combination_dir,
                                                                                       opinion_numbers=3)
        opinions_list = [element["opinion"] for element in opinions_attitudes_importance_list]
        attitudes_description = [element["attitude_description"] for element in opinions_attitudes_importance_list]

        post_prompts = [
            {
                "role": "system",
                "content": self.post_linguistic_instruction["system"]
            },
            {
                "role": "user",
                "content": self.post_linguistic_instruction["user"].format(
                    opinions_list=opinions_list, attitudes_description=attitudes_description
                )
            }
        ]

        discussion_contents = self.openai.get_gpt_response(prompts=post_prompts)

        return discussion_contents

    def join_discussion_linguistic_with_weights(self, combination_dir):
        opinions_attitudes_importance_list = load_latest_opinions_attitudes_importance(combination_dir=combination_dir,
                                                                                       opinion_numbers=3)
        opinions_list = [element["opinion"] for element in opinions_attitudes_importance_list]
        attitudes_description = [element["attitude_description"] for element in opinions_attitudes_importance_list]
        significance_description = [element["significance_description"] for element in
                                    opinions_attitudes_importance_list]

        post_prompts = [
            {
                "role": "system",
                "content": self.post_linguistic_with_weights_instruction["system"]
            },
            {
                "role": "user",
                "content": self.post_linguistic_with_weights_instruction["user"].format(
                    opinions_list=opinions_list, attitudes_description=attitudes_description,
                    significance_description=significance_description
                )
            }
        ]

        discussion_contents = self.openai.get_gpt_response(prompts=post_prompts)

        return discussion_contents

    def analyse_post_without_significance(self, post: str, combination_dir):
        self.status.update(status=500, message="error in analysis")

        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)

        analysis_prompts = [
            {
                "role": "system",
                "content": self.analysis_instruction["system"]
            },
            {
                "role": "user",
                "content": self.analysis_instruction["user"].format(
                    paragraph=post, opinion_list=opinion_list
                )
            }
        ]

        analysis_results = self.openai.get_gpt_response(prompts=analysis_prompts)

        self.status.update(status=200, message="successful analysis")
        return analysis_results

    def analyse_post_with_significance_weights(self, post: str, combination_dir):

        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)

        analysis_with_significance_weights_prompts = [
            {
                "role": "system",
                "content": self.analysis_with_weights_instruction["system"]
            },
            {
                "role": "user",
                "content": self.analysis_with_weights_instruction["user"].format(
                    paragraph=post, opinion_list=opinion_list
                )
            }
        ]

        analysis_with_significance_weights_results = self.openai.get_gpt_response(
            prompts=analysis_with_significance_weights_prompts)

        return analysis_with_significance_weights_results

    def analyse_post_linguistic_without_significance(self, post: str, combination_dir):

        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)

        analysis_linguistic_without_significance_prompts = [
            {
                "role": "system",
                "content": self.analysis_linguistic_instruction["system"]
            },
            {
                "role": "user",
                "content": self.analysis_linguistic_instruction["user"].format(
                    paragraph=post, opinion_list=opinion_list, bases_list=self.bases_list_analysis,
                    modifiers_list=self.modifiers_list
                )
            }
        ]

        analysis_linguistic_without_significance_results = self.openai.get_gpt_response(
            prompts=analysis_linguistic_without_significance_prompts)

        return analysis_linguistic_without_significance_results

    def analyse_post_linguistic_with_significance(self, post: str, combination_dir):
        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)

        analysis_linguistic_with_significance_prompts = [
            {
                "role": "system",
                "content": self.analysis_linguistic_with_weights_instruction["system"]
            },
            {
                "role": "user",
                "content": self.analysis_linguistic_with_weights_instruction["user"].format(
                    paragraph=post, opinion_list=opinion_list, bases_list=self.bases_list_analysis,
                    modifiers_list=self.modifiers_list
                )
            }
        ]

        analysis_linguistic_with_significance_results = self.openai.get_gpt_response(
            prompts=analysis_linguistic_with_significance_prompts)

        return analysis_linguistic_with_significance_results

    def judge_consensus_type(self, combination_dir):
        """
        refine it!
        :param combination_dir:
        :return:
        """
        with open(combination_dir, "r") as f:
            combination_history = json.load(f)
        q1_history = [element for element in combination_history
                      if element["index"] == 1]
        q2_history = [element for element in combination_history
                      if element["index"] == 2]
        q3_history = [element for element in combination_history
                      if element["index"] == 3]

        q1_adjustment = abs(q1_history[0]["attitude"] - q1_history[-1]["attitude"])
        q2_adjustment = abs(q2_history[0]["attitude"] - q2_history[-1]["attitude"])
        q3_adjustment = abs(q3_history[0]["attitude"] - q3_history[-1]["attitude"])

        q1_tolerance_range = self.beta * (1 - q1_history[0]["importance"])
        q2_tolerance_range = (3 * self.beta) * (1 - q2_history[0]["importance"])
        q3_tolerance_range = (3 * self.beta) * (1 - q3_history[0]["importance"])

        if q1_adjustment == 0:
            print("q1 full consensus")
        elif q1_adjustment <= 0.1:
            print("q1 e-full consensus (e=0.1)")
        elif q1_adjustment <= q1_tolerance_range:
            print("q1 strong consensus")
        else:
            print("q1 weak consensus")

        if q2_adjustment == 0:
            print("q2 full consensus")
        elif q2_adjustment <= 0.1:
            print("q2 e-full consensus (e=0.1)")
        elif q2_adjustment <= q2_tolerance_range:
            print("q2 strong consensus")
        else:
            print("q2 weak consensus")

        if q3_adjustment == 0:
            print("q3 full consensus")
        elif q3_adjustment <= 0.1:
            print("q3 e-full consensus (e=0.1)")
        elif q3_adjustment <= q3_tolerance_range:
            print("q3 strong consensus")
        else:
            print("q3 weak consensus")

        return

    def adjust_utilities_with_simple_limitation(self, analysis_dir, combination_dir):
        """
        The function to adjust utilities of the agents upon different opinions.
        The utility adjustment follows both the average attitude response mechanism and the utility adjustment range
        limitation.
        In this version, the limitation is simple: the utility adjustment cannot exceed the range of tolerance of
         each opinion.
        :param analysis_dir:
        :param combination_dir:
        :return:
        """

        # The average attitudes are from the latest recordings.
        # The number is 3 because there are 3 opinions are listed in the experiments
        # This part also needs to be refined in case there are multiple (not a predefined number) opinions included.
        q1_average = calculate_average_utilities(analysis_dir=analysis_dir, combination_dir=combination_dir,
                                                 opinion_index=1, other_agents_number=3)
        q2_average = calculate_average_utilities(analysis_dir=analysis_dir, combination_dir=combination_dir,
                                                 opinion_index=2, other_agents_number=3)
        q3_average = calculate_average_utilities(analysis_dir=analysis_dir, combination_dir=combination_dir,
                                                 opinion_index=3, other_agents_number=3)

        with open(combination_dir, "r") as f2:
            combination_history = json.load(f2)
        data = [copy.deepcopy(item) for item in combination_history[-3:]]
        q1_history = [element for element in combination_history if element["index"] == 1]
        q2_history = [element for element in combination_history if element["index"] == 2]
        q3_history = [element for element in combination_history if element["index"] == 3]

        q1_acceptable_range = self.beta * (1 - q1_history[0]["importance"])
        q2_acceptable_range = self.beta * (1 - q2_history[0]["importance"])
        q3_acceptable_range = self.beta * (1 - q3_history[0]["importance"])

        q1_average_initial_gap = q1_average - q1_history[0]["attitude"]
        q2_average_initial_gap = q2_average - q2_history[0]["attitude"]
        q3_average_initial_gap = q3_average - q3_history[0]["attitude"]

        adjustment_pace_parameter = 1 - (0.1 * (1 - len(q1_history)))

        data[0]["attitude"] = round((q1_average * adjustment_pace_parameter), 2) if abs(
            q1_average_initial_gap) <= q1_acceptable_range else \
            round((q1_history[0]["attitude"] - q1_acceptable_range), 2) if q1_average_initial_gap < 0 else \
            round((q1_history[0]["attitude"] + q1_acceptable_range), 2)
        data[1]["attitude"] = round(q2_average * adjustment_pace_parameter, 2) if abs(
            q2_average_initial_gap) <= q2_acceptable_range else \
            round((q2_history[0]["attitude"] - q2_acceptable_range), 2) if q2_average_initial_gap < 0 else \
            round((q2_history[0]["attitude"] + q2_acceptable_range), 2)
        data[2]["attitude"] = round(q3_average * adjustment_pace_parameter, 2) if abs(
            q3_average_initial_gap) <= q3_acceptable_range else \
            round((q3_history[0]["attitude"] - q3_acceptable_range), 2) if q3_average_initial_gap < 0 else \
            round((q3_history[0]["attitude"] + q3_acceptable_range), 2)

        combination_history = combination_history + data

        with open(combination_dir, "w") as f3:
            json.dump(combination_history, f3, indent=4)

        return

    def adjust_utilities_with_concession_strategy(self, analysis_dir, combination_dir):
        """
        adjust utilities with concession strategies for agents with utilities and the mechanism
        get estimated utilities and significance weights of other agents towards different opinions
        compute relative priority
        compute directional difference
        compute adjusted utilities with concession strategies
        :param analysis_dir:
        :param combination_dir:
        :return:
        """
        other_agents_number = 2
        opinion1_sum_concession_utility = calculate_sum_concession(
            analysis_dir=analysis_dir, combination_dir=combination_dir, opinion_index=1,
            other_agents_number=other_agents_number)
        opinion2_sum_concession_utility = calculate_sum_concession(
            analysis_dir=analysis_dir, combination_dir=combination_dir, opinion_index=2,
            other_agents_number=other_agents_number)
        opinion3_sum_concession_utility = calculate_sum_concession(
            analysis_dir=analysis_dir, combination_dir=combination_dir, opinion_index=3,
            other_agents_number=other_agents_number)

        #  need to be improved no beta (beta=1)
        opinion1_average_concession_utility = opinion1_sum_concession_utility / other_agents_number
        opinion2_average_concession_utility = opinion2_sum_concession_utility / other_agents_number
        opinion3_average_concession_utility = opinion3_sum_concession_utility / other_agents_number

        with open(combination_dir, "r") as f2:
            combination_history = json.load(f2)
        data = [copy.deepcopy(item) for item in combination_history[-3:]]
        opinion1_adjusted_utility = data[0]["attitude"] + opinion1_average_concession_utility
        data[0]["attitude"] = round(opinion1_adjusted_utility, 2)
        opinion2_adjusted_utility = data[1]["attitude"] + opinion2_average_concession_utility
        data[1]["attitude"] = round(opinion2_adjusted_utility, 2)
        opinion3_adjusted_utility = data[2]["attitude"] + opinion3_average_concession_utility
        data[2]["attitude"] = round(opinion3_adjusted_utility, 2)

        combination_history = combination_history + data

        with open(combination_dir, "w") as f3:
            json.dump(combination_history, f3, indent=4)

        return

    def adjust_utilities_llm_with_reference(self, analysis_dir, combination_dir):
        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)
        latest_opinions_attitudes_importance = load_latest_opinions_attitudes_importance(
            combination_dir=combination_dir, opinion_numbers=3)
        utilities_list = [element["attitude"] for element in latest_opinions_attitudes_importance]
        s_weights_list = [element["importance"] for element in latest_opinions_attitudes_importance]
        average_utilities_list = []
        for i in range(1, len(opinion_list) + 1):  # 1, opinion_number + 1
            average_attitude = calculate_average_utilities(analysis_dir=analysis_dir, combination_dir=combination_dir,
                                                           opinion_index=i, other_agents_number=3)
            average_utilities_list.append(average_attitude)

        adjust_utility_llm_with_reference_prompts = [
            {
                "role": "system",
                "content": self.utility_adjustment_with_references_instruction["system"]
            },
            {
                "role": "user",
                "content": self.utility_adjustment_with_references_instruction["user"].format(
                    opinion_list=opinion_list, utilities_list=utilities_list, s_weights_list=s_weights_list,
                    average_utilities_list=average_utilities_list, beta=self.beta
                )
            }
        ]

        adjusted_utilities = self.openai.get_gpt_response(prompts=adjust_utility_llm_with_reference_prompts)
        adjusted_utilities = eval(adjusted_utilities)
        adjusted_utilities = list(adjusted_utilities.values())  # [u_1, u_2, u_3]

        with open(combination_dir, "r") as f1:
            combination_history = json.load(f1)
        for i in range(0, len(adjusted_utilities)):
            latest_opinions_attitudes_importance[i]["attitude"] = adjusted_utilities[i]
        combination_history = combination_history + latest_opinions_attitudes_importance

        with open(combination_dir, "w") as f2:
            json.dump(combination_history, f2, indent=4)

        return

    def adjust_utilities_llm_with_reference_and_concession(self, analysis_dir, combination_dir):
        """
        refine this function to make agents adjust utilities one by one since we need to provide CoT and few-shot
        learning examples to agents. Combine the adjusted utilities after the adjustment.
        :param analysis_dir:
        :param combination_dir:
        :return:
        """
        latest_opinions_attitudes_importance = load_latest_opinions_attitudes_importance(
            combination_dir=combination_dir, opinion_numbers=3)
        with open(analysis_dir, "r") as analysis_file:
            analysis_data = json.load(analysis_file)
        opinion_number_temp = 3
        other_agent_number_temp = 2
        analysis_data = analysis_data[-(opinion_number_temp * other_agent_number_temp):]
        separated_analysis_data = separate_analysis_results_by_opinion_index(analysis_data=analysis_data)
        # the length of separated_analysis_data should be the same as opinion_number_temp and len(
        # latest_opinions_attitudes_importance)
        if len(separated_analysis_data) != opinion_number_temp:
            print("check separate array function!")

        adjusted_utility_list = []

        for i, latest in enumerate(latest_opinions_attitudes_importance):
            other_agent_same_opinion_list = separated_analysis_data[i]
            if other_agent_same_opinion_list[0]["index"] != latest["index"]:
                print("check separate array function! (index and ordering)")

            relative_priority_1 = round((other_agent_same_opinion_list[0]["significance"]/(latest["importance"] + 1)), 2)
            relative_priority_2 = round((other_agent_same_opinion_list[1]["significance"]/(latest["importance"] + 1)), 2)

            agent_information_block = {
                "opinion": latest["opinion"],
                "your_current_utility": latest["attitude"],
                f"current_utility_of_{other_agent_same_opinion_list[0]['agent name']}":
                    other_agent_same_opinion_list[0]["attitude"],
                f"relative_priority_with_{other_agent_same_opinion_list[0]['agent name']}":
                    relative_priority_1,
                f"current_utility_of_{other_agent_same_opinion_list[1]['agent name']}":
                    other_agent_same_opinion_list[1]["attitude"],
                f"relative_priority_with_{other_agent_same_opinion_list[1]['agent name']}":
                    relative_priority_2
            }

            adjust_utilities_llm_with_reference_and_concession_prompts = [
                {
                    "role": "system",
                    "content": self.utility_adjustment_with_references_and_concession_instruction["system"]
                },
                {
                    "role": "user",
                    "content": self.utility_adjustment_with_references_and_concession_instruction["user"].format(
                        agents_opinion_information_block=agent_information_block
                    )
                }
            ]

            adjusted_utility = self.openai.get_gpt_response(
                prompts=adjust_utilities_llm_with_reference_and_concession_prompts)
            adjusted_utility = json.loads(adjusted_utility)
            latest["attitude"] = adjusted_utility["adjusted_utility"]
            adjusted_utility_list.append(latest)

        with open(combination_dir, "r") as f1:
            combination_history = json.load(f1)
        combination_history = combination_history + adjusted_utility_list

        with open(combination_dir, "w") as f2:
            json.dump(combination_history, f2, indent=4)

        return

    def adjust_attitudes_llm_with_description(self, analysis_dir, combination_dir):
        opinion_list = load_opinions(combination_dir=combination_dir, opinion_numbers=3)
        latest_opinions_attitudes_importance = load_latest_opinions_attitudes_importance(
            combination_dir=combination_dir, opinion_numbers=3)
        attitudes_descriptions_list = [element["attitude_description"] for element in
                                       latest_opinions_attitudes_importance]
        significance_descriptions_list = [element["significance_description"] for element in
                                          latest_opinions_attitudes_importance]
        others_attitude_descriptions_per_opinion_2d_list = combine_others_linguistic_attitudes(
            analysis_dir=analysis_dir, opinion_number=3)
        acceptance_description = self.beta_to_acceptability_descriptions()

        adjust_preference_llm_with_descriptions = [
            {
                "role": "system",
                "content": self.attitude_adjustment_with_description_instruction["system"]
            },
            {
                "role": "user",
                "content": self.attitude_adjustment_with_description_instruction["user"].format(
                    opinion_list=opinion_list, your_attitude_descriptions=attitudes_descriptions_list,
                    your_significance_descriptions=significance_descriptions_list,
                    your_acceptance_description=acceptance_description,
                    others_attitude_descriptions_per_opinion=others_attitude_descriptions_per_opinion_2d_list,
                    bases_list=self.bases_list_adjustment,
                    modifiers_list=self.modifiers_list
                )
            }
        ]

        adjusted_attitudes_descriptions = self.openai.get_gpt_response(prompts=adjust_preference_llm_with_descriptions)
        adjusted_attitudes_descriptions = eval(adjusted_attitudes_descriptions)
        # [attitude_1, attitude_2, ...]
        adjusted_attitudes_descriptions = list(adjusted_attitudes_descriptions.values())

        with open(combination_dir, "r") as f1:
            combination_history = json.load(f1)
        for i in range(0, len(adjusted_attitudes_descriptions)):
            latest_opinions_attitudes_importance[i]["attitude_description"] = adjusted_attitudes_descriptions[i]
        combination_history = combination_history + latest_opinions_attitudes_importance

        with open(combination_dir, "w") as f2:
            json.dump(combination_history, f2, indent=4)

        return

    def adjust_attitudes_llm_with_description_and_concession(self, analysis_dir, combination_dir):
        """
        refine this function to make agents adjust attitudes one by one since we need to provide CoT and few-shot
        learning examples to the prompts. Combine the adjusted attitudes after adjustment
        :param analysis_dir:
        :param combination_dir:
        :return:
        """
        latest_opinions_attitudes_importance = load_latest_opinions_attitudes_importance(
            combination_dir=combination_dir, opinion_numbers=3)
        with open(analysis_dir, "r") as analysis_file:
            analysis_data = json.load(analysis_file)
        opinion_number_temp = 3
        other_agent_number_temp = 2
        analysis_data = analysis_data[-(opinion_number_temp * other_agent_number_temp):]
        separated_analysis_data = separate_analysis_results_by_opinion_index(analysis_data=analysis_data)
        # the length of separated_analysis_data should be the same as opinion_number_temp and len(
        # latest_opinions_attitudes_importance)
        if len(separated_analysis_data) != opinion_number_temp:
            print("check separate array function!")

        adjusted_attitude_list = []

        for i, latest in enumerate(latest_opinions_attitudes_importance):
            other_agent_same_opinion_list = separated_analysis_data[i]
            if other_agent_same_opinion_list[0]["index"] != latest["index"]:
                print("check separate array function! (index and ordering)")

            agent_information_block = {
                "opinion": latest["opinion"],
                "your_current_attitude": latest["attitude_description"],
                "your_significance_level": latest["significance_description"],
                f"current_attitude_of_{other_agent_same_opinion_list[0]['agent name']}": other_agent_same_opinion_list[0]["attitude"],
                f"significance_level_of_{other_agent_same_opinion_list[0]['agent name']}": other_agent_same_opinion_list[0]["significance"],
                f"current_attitude_of_{other_agent_same_opinion_list[1]['agent name']}": other_agent_same_opinion_list[1]["attitude"],
                f"significance_level_of_{other_agent_same_opinion_list[1]['agent name']}": other_agent_same_opinion_list[1]["significance"]
            }

            adjust_preference_llm_with_descriptions_and_concession_prompts = [
                {
                    "role": "system",
                    "content": self.attitude_adjustment_with_description_and_concession_instruction["system"]
                },
                {
                    "role": "system",
                    "content": self.attitude_adjustment_with_description_and_concession_instruction["user"].format(
                        agent_information_block=agent_information_block,
                        bases_list=self.bases_list_adjustment,
                        modifiers_list=self.modifiers_list
                    )
                }
            ]

            adjusted_attitudes_description = self.openai.get_gpt_response(
                prompts=adjust_preference_llm_with_descriptions_and_concession_prompts)
            if adjusted_attitudes_description.startswith("```"):
                adjusted_attitudes_description = adjusted_attitudes_description.replace("```", "")
            if adjusted_attitudes_description.startswith("json"):
                adjusted_attitudes_description = adjusted_attitudes_description[len("json"):].strip()
            adjusted_attitudes_description = json.loads(adjusted_attitudes_description)
            latest["attitude_description"] = adjusted_attitudes_description["adjusted_attitude_description"]
            adjusted_attitude_list.append(latest)

        with open(combination_dir, "r") as f1:
            combination_history = json.load(f1)
        combination_history = combination_history + adjusted_attitude_list

        with open(combination_dir, "w") as f2:
            json.dump(combination_history, f2, indent=4)

        return
