__author__ = "Dong Yihan"

import os.path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Load questions from a JSON file
def load_questions(json_file):
    with open(json_file, 'r') as f:
        questions = json.load(f)
    return questions


questions = load_questions('./questionnaire/questionnaire_test.json')


# Mapping the slider value to the textual representation
def get_textual_value(value):
    if value == -2:
        return "Strongly Disagree"
    elif value == -1:
        return "Disagree"
    elif value == 0:
        return "Neutral"
    elif value == 1:
        return "Agree"
    elif value == 2:
        return "Strongly Agree"
    elif -2 < value < -1:
        return f"Strongly Disagree-{(value + 2) / 1}-Disagree"
    elif -1 < value < 0:
        return f"Disagree-{(value + 1) / 1}-Neutral"
    elif 0 < value < 1:
        return f"Neutral-{value / 1}-Agree"
    elif 1 < value < 2:
        return f"Agree-{(value - 1) / 1}-Strongly Agree"


# Save the answers to a JSON file
def save_answer(question_id, answer_text, answer_value):
    answer_data = {
        "question_id": question_id,
        "answer_text": answer_text,
        "answer_value": answer_value
    }

    if os.path.exists("./questionnaire/answers_test.json"):
        # 事前に回答のjsonファイルに[]を入れてください
        with open("./questionnaire/answers_test.json", "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(answer_data)

    with open("./questionnaire/answers_test.json", "w") as f:
        json.dump(data, f, indent=4)


@app.get("/", response_class=HTMLResponse)
async def read_question(request: Request, question_index: int = 0):
    if question_index < len(questions):
        question = questions[question_index]["text"]
        question_id = questions[question_index]["id"]
        return templates.TemplateResponse("questionnaire_test.html",
                                          {"request": request, "question": question, "question_id": question_id})
    else:
        return templates.TemplateResponse("finish_questionnaire_test.html",
                                          {"request": request})


@app.post("/submit/")
async def submit_answer(request: Request, question_id: int = Form(...), answer: str = Form(...)):
    print(f"User's response to question {question_id}: {answer}")
    answer_value = float(answer)
    answer_text = get_textual_value(answer_value)
    save_answer(question_id, answer_text, answer_value)
    return await read_question(request, question_index=question_id)


# Generate all possible pairs of question indices
def generate_question_pairs(questions: list):
    pairs = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            pairs.append((questions[i], questions[j]))
    return pairs


question_pairs = generate_question_pairs(questions)


# Save the comparison result to a JSON file
def save_comparison_result(first_question_id, second_question_id, selected_question_id,
                           base_question, importance_level):
    comparison_result = {
        "first_question_id": first_question_id,
        "second_question_id": second_question_id,
        "selected_question_id": selected_question_id,
        "base_question_id": base_question,
        "importance_level": importance_level
    }

    try:
        if os.path.exists("./questionnaire/comparison_test.json"):
            with open("./questionnaire/comparison_test.json", "r") as f:
                data = json.load(f)
        else:
            data = []
    except json.JSONDecodeError:
        data = []

    data.append(comparison_result)

    with open("./questionnaire/comparison_test.json", "w") as f:
        json.dump(data, f, indent=4)


@app.get("/compare/", response_class=HTMLResponse)
async def read_root(request: Request):
    return await display_pair(request, pair_index=0)


@app.get("/display_pair/", response_class=HTMLResponse)
async def display_pair(request: Request, pair_index: int):
    if pair_index < len(question_pairs):
        first_question_id = question_pairs[pair_index][0]["id"]
        second_question_id = question_pairs[pair_index][1]["id"]
        first_question_text = question_pairs[pair_index][0]["text"]
        second_question_text = question_pairs[pair_index][1]["text"]
        return templates.TemplateResponse("comparison_test.html", {
            "request": request,
            "first_question_id": first_question_id,
            "second_question_id": second_question_id,
            "first_question_text": first_question_text,
            "second_question_text": second_question_text,
            "pair_index": pair_index
        })
    else:
        return HTMLResponse(content="You have viewed all question pairs. Thank you!", status_code=200)


@app.post("/submit_selection/", response_class=HTMLResponse)
async def submit_selection(request: Request, pair_index: int = Form(...), selected_question: int = Form(...),
                           importance_level: int = Form(...)):
    # Save the user's selection
    first_question_id = question_pairs[pair_index][0]["id"]
    second_question_id = question_pairs[pair_index][1]["id"]
    if selected_question == first_question_id:
        base_question = second_question_id
    else:
        base_question = first_question_id
    save_comparison_result(first_question_id, second_question_id, selected_question, base_question, importance_level)

    # Move to the next pair
    return await display_pair(request, pair_index=pair_index + 1)
