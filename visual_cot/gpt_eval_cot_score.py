import os
import openai
import time
from tqdm import tqdm
import json
import argparse
import re
import requests
from glob import glob


BASE_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: %s
standard answer: %s
model's answer: %s
"""

API_KEY = os.environ['OPENAI_API_KEY']


def make_request_openai(content, extra_args={}):
    headers = {}
    headers['Content-Type']='application/json'
    retry_times = 3
    while retry_times > 0:
        try:
            data = {}
            data['model']= "gpt-3.5-turbo-1106"
            data['messages'] = [{"role":"system","content": BASE_PROMPT}, {"role": "user", "content":content}]
            for key in extra_args:
                data[key] = extra_args[key]
            headers['Authorization'] = API_KEY
            r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=60)
            response = r.json()
            response = response['choices'][0]['message']['content']
            return response
        except Exception as e:
            print(e)
            time.sleep(1)
        finally:
            retry_times -= 1
    return 'unknown'


def get_score(question_text, gt_answer_text, pred_answer_text):
    content = PROMPT % (question_text, gt_answer_text, pred_answer_text)
    ret = make_request_openai(content)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    args = parser.parse_args()
    result_files = glob(f'{args.result_dir}/*.json')

    for result_file in result_files:
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
