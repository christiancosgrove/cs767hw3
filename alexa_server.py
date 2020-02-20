from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
import os
import random
import subprocess

app = Flask(__name__)
ask = Ask(app, "/")

@ask.intent("AskDeepPavlov", mapping={'user_input':'raw_input'})
def response_from_model():
    response = subprocess.check_output('user_input').decode()
    return question(response)

if __name__ == '__main__':
    app.run(debug=True)
    cmd = 'python -m parlai.scripts.interactive -mf zoo:convai2/seq2seq/convai2_self_seq2seq_model -m legacy:seq2seq:0'
    # cmd = 'python3 examples/interactive.py -m hred/hred -mf test_hred.checkpoint.checkpoint'
    os.system(cmd)