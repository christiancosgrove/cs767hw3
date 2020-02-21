from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
import os
import random
import subprocess

app = Flask(__name__)
ask = Ask(app, "/")

@ask.launch
def welcome():
	cmd = 'python3 examples/interactive.py -m hred/hred -mf test_hred.checkpoint.checkpoint'
	os.system(cmd)
	#welcome_message = 'Hello there, would you like the news?'
	#return question(welcome_message)
	model_setup = "model finished setting up"
	return statement(model_setup)

@ask.intent("AskDeepPavlov", mapping={'user_input':'raw_input'})
def response_from_model():
	while True:
		response = subprocess.check_output('user_input').decode()
		response = response[7:] # Should get rid of [HRED]: 
		return question(response).reprompt("This is a debugger to check we're still in HRED")

if __name__ == '__main__':
	app.run(debug = True)