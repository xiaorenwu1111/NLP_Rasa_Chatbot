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

logger =logger.getLogger(__name__)

def train_dialogue(domain_file="C:/Users/liyufang/Desktop/Weatherbot_Tutorial-master/Video files/weather_domain.yml",
                    model_path='C:/Users/liyufang/Desktop/Weatherbot_Tutorial-master/Video files/models/dialogue',
                    training_data_file="C:/Users/liyufang/Desktop/Weatherbot_Tutorial-master/Video files/data/stories.md")
                

    agent=Agent(domain_file,polices=[MemoizationPolicy(),KerasPolicy()])

    agent.train(
        training_data_file,
        max_history = 3 ,
        epochs = 300 ,
        batch_size =50 ,
        validation_split = 0.2,
        augmentation_factor=50)

    agent.persist(model_path)
    return agent 

def run_weather_bot(serve_forever=True):
    interpreter=RasaNLUInterpreter('C:/Users/liyufang/Desktop/Weatherbot_Tutorial-master/Video files/models/nlu/default/weathernlu')
    agent=Agent.load('C:/Users/liyufang/Desktop/Weatherbot_Tutorial-master/Video files/models/nlu/default/weathernlu',interpreter=interpreter)


    if serve_forever:
        agent.handle_channel_channel(ConsoleInputChannel())
    return agent 

if __name__ == '__main__':
    train_dialogue()
    run_weather_bot()


