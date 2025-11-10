# 

import paho.mqtt.client as mqtt
import time

broker = "10.8.4.11"
port = 1883
topic = 'topic/uicdatest1'

client = mqtt.Client()
client.connect(broker, port, 60)

while True:
    message = input('Enter a message to send: ')
    client.publish(topic, message)
    print(f'Sent: {message}')
    time.sleep(0.5)
