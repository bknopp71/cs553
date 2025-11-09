import paho.mqtt.client as mqtt
import json, time

BROKER = "192.168.1.11"
TOPIC = "robot/chat"
CLIENT_ID = "robot2_slave"
STATE = 0
FINISHED = False

def on_connect(client, userdata, flags, rc):
    print("Slave connected.")
    client.subscribe(TOPIC)
    # Always announce ready state
    client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "state": STATE}))

def on_message(client, userdata, msg):
    global STATE, FINISHED
    data = json.loads(msg.payload.decode())
    sender = data.get("sender")
    s = data.get("state", 0)
    stop = data.get("stop", False)

    if sender == CLIENT_ID or FINISHED:
        return

    if stop:
        print("Slave stopping.")
        FINISHED = True
        client.disconnect()
        return

    # Follow master's state
    if s >= STATE:
        STATE = s
        print(f"Slave -> State {STATE}")
        time.sleep(0.5)
        client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "state": STATE}))

client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_forever()