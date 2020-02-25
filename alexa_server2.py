from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
import os
import random
import subprocess

import subprocess
from subprocess import PIPE
from time import sleep

global p

app = Flask(__name__)
ask = Ask(app, "/")
@ask.launch
def welcome():
	model_setup = "connected"
	return statement(model_setup)

@ask.intent("AskDeepPavlov", mapping={'user_input':'raw_input'})
def response_from_model(user_input):
	while True:
		p.stdin.write(b'a line\n')
		p.stdin.flush()
		response = p.stdout.readline()
		response = subprocess.check_output(user_input).decode()
		response = response[7:] # Should get rid of [HRED]: 
		return question(response).reprompt("This is a debugger to check we're still in HRED")


def start_model():
	p = subprocess.Popen(["python","examples/interactive.py","-m" ,"hred", "-mf","test_hred.checkpoint"], stdin=PIPE, stdout=PIPE, bufsize=1)
	sleep(10)
	p.stdout.flush()

if __name__ == '__main__':
	start_model()
	app.run(debug = True)