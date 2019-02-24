Building a chatbot with Rasa Nlu and Rasa Core
===============================================
一、文件准备部分
-----------------
```python
_rasa_Weatherbot/
├── requirements.txt   #安装环境依赖的packages
├── chat_detection
├── data
│   ├──data.json  # train data json format (nlu) 
│   ├── mobile_raw_data.txt # train data raw
│   ├── mobile_story.md # toy dialogue train data 
│   └── total_word_feature_extractor.dat # pretrained mitie word vector 
├──weather_domain.yml # rasa core
├── __init__.py
├── INSTALL.md
├──config_spacy.json  # rasa nlu config file
├── mobile_domain.yml # rasa core config file
├── projects # pretrained models
│   ├── dialogue
│   └── ivr_nlu
├── README.md
├── tools # tools of data process
└── train.sh # train script of rasa nlu
```

二、操作流程
------------
 (一)、准备工作-环境搭建及依赖安装
 
1.安装依赖环境
```
python pip install –r requirements.txt
```

2.download language model
```python
Python –m spacy download en  (#down load English model)
```

3. install  Node.js   #安装npm，让后面下载rasa-nlu-trainner更加流畅
 

4.安装rasa-nlu-trainner  #安装模型训练器
```python
npm  i –g rasa-nlu-trainner
```


（二）模型训练<br>
一）Training a rasa-nlu model<br>
 
1.创建data文件夹<br>
```shell
mkdir data
cd data
echo ‘nlu_data’ >data.json
```

2.准备训练数据-data.json file<br>
Method 1: 自己编写training data<br>
```python
{
  "rasa_nlu_data": {
    "common_examples": [
      {
        "text": "Hello",
        "intent": "greet",
        "entities": []
      },
      {
        "text": "goodbye",
        "intent": "goodbye",
        "entities": []
      },
      {
        "text": "What's the weather in Berlin at the moment?",
        "intent": "inform",
        "entities": [
          {
            "start": 22,
            "end": 28,
            "value": "Berlin",
            "entity": "location"
          }
        ]
      }
```


Method 2:<br>
本地键入command 访问data.json,在网页端编写训练数据<br>
```python
rasa-nlu-trainer
```
 
3.创建及编写配置文件-cofig_spacy.json<br>
（1）创建文件<br>
```python
echo ‘config’ > config_spacy.json
```

（2）编写配置文件<br>
```python
{
  "pipeline":"spacy_sklearn",
  "path":"./models/nlu",
  "data":"./data/data.json"
}
```

4.模型训练<br>
（1）创建python script<br>
```python
echo ‘nlu_model’ > nlu_model.py 
```

（2）编写python script<br>
```python
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

def train_nlu(data, config, model_dir):
	training_data = load_data(data)
	trainer = Trainer(RasaNLUConfig(config))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'weathernlu')
	
	
if __name__ == '__main__':
	train_nlu('./data/data.json', 'config_spacy.json', './models/nlu')
	run_nlu()
```

（3）训练模型<br>
```python
python nlu_model.py
```

(4)重新修改python script ，训练模型后显示打印训练结果<br>
```python
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter  #添加的部分

def train_nlu(data, config, model_dir):
	training_data = load_data(data)
	trainer = Trainer(RasaNLUConfig(config))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'weathernlu')
	
def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/weathernlu', RasaNLUConfig('config_spacy.json'))
	print(interpreter.parse("I am planning my holiday to Lithuania. I wonder what is the weather out there."))
	
if __name__ == '__main__':
	#train_nlu('./data/data.json', 'config_spacy.json', './models/nlu')
	run_nlu()
```

（5）再次训练模型<br>
```python
python nlu_model.py
```

5.output<br>
```python
_rasa_nlu output /
├── requirements.txt   #安装环境依赖的packages
├── chat_detection
├── models/nlu/default/weathernlu
│   ├──crf_model.pkl   
│   ├──intent_classifier.pkl 
│   ├──metadata.json 
│   └──training_data.json 
```



二）Rasa Core<br> 
 
 


1.数据准备<br>
（1）创建domain file<br>
```python
echo ‘domain’>weather_domain.yml
```

（2）编辑domain file<br>
```python
slots:
  location:
    type: text   #what data type of this model


intents:
 - greet
 - goodbye
 - inform


entities:
 - location

templates:      #the response of the of 
  utter_greet:
    - 'Hello! How can I help?'
  utter_goodbye:
    - 'Talk to you later.'
    - 'Bye bye :('
  utter_ask_location:
    - 'In what location?'


actions:
 - utter_greet
 - utter_goodbye
 - utter_ask_location
 - actions.ActionWeather
```

（3）创建action script<br>
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_core.actions.action import Action
from rasa_core.events import SlotSet

class ActionWeather(Action):
	def name(self):
		return 'action_weather'
		
	def run(self, dispatcher, tracker, domain):
		from apixu.client import ApixuClient
		api_key = '...' #your apixu key 通过API网站注册获得APIkey
		client = ApixuClient(api_key)
		
		loc = tracker.get_slot('location')
		current = client.getCurrentWeather(q=loc)
		
		country = current['location']['country']
		city = current['location']['name']
		condition = current['current']['condition']['text']
		temperature_c = current['current']['temp_c']
		humidity = current['current']['humidity']
		wind_mph = current['current']['wind_mph']

		response = """It is currently {} in {} at the moment. The temperature is {} degrees, the humidity is {}% and the wind speed is {} mph.""".format(condition, city, temperature_c, humidity, wind_mph)
						
		dispatcher.utter_message(response)
		return [SlotSet('location',loc)]
```

API网站申请APIXU<br>
 

 

（4）将创建的actions添加到weather_domain.yml中<br>
 


 

（5）创建story data 以及编辑story data<br>
```python
cd data
echo ‘stories’ > stories.md

* greet
    - utter_greet
* inform
    - utter_ask_location
* inform{"location": "italy"}
    - slot{"location": "italy"}
    - action_weather
    - slot{"location": "italy"}
* goodbye
    - utter_goodbye
- export

```

2.训练脚本<br>
（1）train_int.py<br>
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

if __name__ == '__main__':
	logging.basicConfig(level='INFO')
	
	training_data_file = './data/stories.md'
	model_path = './models/dialogue'
	
	agent = Agent('weather_domain.yml', policies = [MemoizationPolicy(), KerasPolicy()])
	
	agent.train(
			training_data_file,
			augmentation_factor = 50,
			max_history = 2,
			epochs = 500,
			batch_size = 10,
			validation_split = 0.2)
			
	agent.persist(model_path)
```


(2)训练模型<br>
```python
python train_int.py
```

（3）训练结果<br>
 


（4）train_online,py<br>
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter

logger = logging.getLogger(__name__)


def run_weather_online(input_channel, interpreter,
                          domain_file="weather_domain.yml",
                          training_data_file='data/stories.md'):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')
    run_weather_online(ConsoleInputChannel(), nlu_interpreter)
```

(5)online training<br>
```python
python train_online,py
```

（6）创建dialogue 模型训练script<br> 
```python
echo ‘dialogue’>dialogue_management_model.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter

logger = logging.getLogger(__name__)

def train_dialogue(domain_file = 'weather_domain.yml',
					model_path = './models/dialogue',
					training_data_file = './data/stories.md'):
					
	agent = Agent(domain_file, policies = [MemoizationPolicy(), KerasPolicy()])
	
	agent.train(
				training_data_file,
				max_history = 3,
				epochs = 300,
				batch_size = 50,
				validation_split = 0.2,
				augmentation_factor = 50)
				
	agent.persist(model_path)
	return agent
	
def run_weather_bot(serve_forever=True):
	interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')
	agent = Agent.load('./models/dialogue', interpreter = interpreter)
	
	if serve_forever:
		agent.handle_channel(ConsoleInputChannel())
		
	return agent
	
if __name__ == '__main__':
	train_dialogue()
	run_weather_bot()
```


(7)运行脚本<br>
```python
python dialogue_management_model.py
```


3.通过slack+chatrot<br>
（1）登录slack页面<br>
 

（2）create slack API<br>
 

（3）add function / add botuser<br>
 


(4) install the app in work place<br>
 

 

(5) write a script for rasa robot app<br>
```python
Create a python script for rasa slack connector
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from builtins import str
from flask import Blueprint, request, jsonify

from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class SlackBot(OutputChannel):
	def __init__(self, slack_verification_token, channel):
		self.slack_verification_token = slack_verification_token
		self.channel = channel
		
	def send_text_message(self, recipient_id, message):
		from slackclient import SlackClient
		text = message
		recipient = recipient_id
		
		CLIENT = SlackClient(self.slack_verification_token)
		CLIENT.api_call('chat.postMessage', channel = self.channel, text = text, as_user = True)
		




class SlackInput(HttpInputComponent):
	def __init__(self, slack_dev_token, slack_verification_token, slack_client, debug_mode):
		self.slack_dev_token = slack_dev_token
		self.debug_mode = debug_mode
		self.slack_client = slack_client
		self.slack_verification_token = slack_verification_token
		
	def blueprint(self, on_new_message):
		from flask import Flask, request, Response
		slack_webhook = Blueprint('slack_webhook', __name__)
		
		@slack_webhook.route('/', methods = ['GET'])
		def health():
			return jsonify({'status':'ok'})
			
		@slack_webhook.route('/slack/events', methods = ['POST'])
		def event():
			if request.json.get('type') == 'url_verification':
				return request.json.get('challenge'), 200
				
			if request.json.get('token') == self.slack_client and request.json.get('type') == 'event_callback':
				data = request.json
				messaging_events = data.get('event')
				channel = messaging_events.get('channel')
				user = messaging_events.get('user')
				text = messaging_events.get('text')
				bot = messaging_events.get('bot_id')
				if bot == None:
					on_new_message(UserMessage(text, SlackBot(self.slack_verification_token, channel)))
					
			return Response(), 200
			
		return slack_webhook

```

(6)write run app python script<br> 
```python
from rasa_core.channels import HttpInputChannel
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_slack_connector import SlackInput


nlu_interpreter = RasaNLUInterpreter('./models/nlu/default/weathernlu')
agent = Agent.load('./models/dialogue', interpreter = nlu_interpreter)

input_channel = SlackInput('xoxp...', #app verification token
			'xoxb...', # bot verification token
			'...', # slack verification token
True)
#we can find the token from slack web ,such as below page

agent.handle_channel(HttpInputChannel(5004, '/', input_channel))

```
 


(7)run script and run app<br>

```python
python  run_app.py
```
 

这种情况我们需要重新链接一下local 的robot 以及网页版的slack <br>
具体操作如下：<br>
 
 
Back to slack<br>
 
 
 

 

 
Add events<br>
 


Say hello to the chat app<br>


参考链接：<br>
https://jpboost.com/2018/02/06/creating-a-chatbot-with-rasa-nlu-and-rasa-core/

