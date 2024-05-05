import os
import openai
import time
from tqdm import tqdm
import json
import argparse
import re
import requests


BASE_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: %s
standard answer: %s
model's answer: %s
"""

API_KEY = os.environ['OPENAI_API_KEY']
GPT_EVAL_MODEL_NAME = "gpt-3.5-turbo-1106"
API_TYPE = os.getenv("API_TYPE", "openai")

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
# API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
}


def get_eval(content: str, max_tokens=100, retries: int = 5):
    global headers
    messages = [
        {
            "role": "system",
            "content": BASE_PROMPT,
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"]
            if content != "":
                return content
            break  # If successful, break out of the loop

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(5)
            else:  # If this was the last attempt, log and return empty
                print(f"All {retries} attempts failed. Last error message: {e}")
                return ""
    return ""


def get_score(question_text, gt_answer_text, pred_answer_text):
    content = PROMPT % (question_text, gt_answer_text, pred_answer_text)
    ret = get_eval(content)
    ret = ret.lower()
    if 'score' not in ret:
        return 0.0
    res = re.findall(r'score: ([\d\.]+)', ret)
    if len(res) != 1:
        return 0.0
    res = float(res[0])
    if res > 1.0:
        res = 1
    if res < 0.0:
        res = 0
    time.sleep(1)    # sleep for 1 second after a successful request to avoid high frequency
    return res


if __name__ == "__main__":
    trial = get_eval('Who are you?')
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str)
    args = parser.parse_args()
    result_file = args.result_file
    print(f"Processing {result_file}", flush=True)
    scores = []
    with open(result_file, 'r') as f:
        data = json.load(f)
    for data_sample in tqdm(data):
        score = get_score(data_sample['question'], data_sample['gt'], data_sample['answer'])
        scores.append(score)
        data_sample['score'] = score

    print(f'The avg score on {result_file} is: {sum(scores) / len(scores)}', flush=True)

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=4)
