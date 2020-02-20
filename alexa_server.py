from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
from parlai.scripts.interactive import setup_args, interactive
import random

app = Flask(__name__)
ask = Ask(app, "/")

@ask.intent("AskDeepPavlov", mapping={'input':'raw_input'})
def response_from_model():
	
    # random.seed(42)
    # parser = setup_args()
    # opt = parser.parse_args()
    # interactive(opt, print_parser=parser)

    
	return statement(response)


if __name__ == '__main__':
	app.run(debug=True)