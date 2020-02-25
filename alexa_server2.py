from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
import os
import random
import subprocess

import subprocess
from subprocess import PIPE
from time import sleep

app = Flask(__name__)
ask = Ask(app, "/")


logging.getLogger('flask_ask').setLevel(logging.DEBUG)

@ask.launch
def welcome():
	model_setup = "connected"
	return question(model_setup)

@ask.intent("AskDeepPavlov", mapping={'user_input':'raw_input'})
def response_from_model(user_input):
	print("USER INPUT IS ", user_input)
	
	target = 'Enter Your Message'
	m = ''
	while m[:len(target)] != target and p.poll() is None:
		p.stdin.write(b'a line\n')
		p.stdin.flush()
		m = p.stdout.readline().decode()
	
	p.stdin.write(user_input.encode())
	p.stdin.flush()
	response = p.stdout.readline().decode()
	print('RESPONSE', response)
	response = response[7:] # Should get rid of [HRED]: 
	return question(response).reprompt("This is a debugger to check we're still in HRED")


def start_model():
	global p
	p = subprocess.Popen(["python","examples/interactive.py","-m" ,"hred", "-mf","test_hred.checkpoint"], stdin=PIPE, stdout=PIPE, bufsize=1)
	sleep(10)
	p.stdout.flush()
	print('flushed!')#, p.stdout.read())

if __name__ == '__main__':
	start_model()
	app.run(debug = True)