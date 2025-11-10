import paho.mqtt.client as mqtt
import json, time

#BROKER = "192.168.1.11"
BROKER = "10.8.4.35"
TOPIC = "robot/chat"
CLIENT_ID = "robot1_master"
STATE = 0
MAX_STATE = 6
FINISHED = False

def on_connect(client, userdata, flags, rc):
    print("Master connected.")
    client.subscribe(TOPIC)
    # Always announce current state
    client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "state": STATE}))
    print(f"Master state {STATE} announced.")

def on_message(client, userdata, msg):
    global STATE, FINISHED
    data = json.loads(msg.payload.decode())
    sender = data.get("sender")
    s = data.get("state", 0)
    stop = data.get("stop", False)

    if sender == CLIENT_ID or FINISHED:
        return

    if stop:
        print("Master received stop. Disconnecting.")
        FINISHED = True
        client.disconnect()
        return

    # When slave matches, advance
    if s == STATE:
        STATE += 1

        if STATE == 1:
            input("Robot 1 finished bulldozing, Press Enter to finish state 1...")
            # put logic to take picture and bulldoze
        if STATE == 2:
            print("Waiting for robot 2 to finish bulldozing")
            # Do nothing, robot 2 is bulldozing
        if STATE == 3:
            input("Robot 1 take picture------------------Press Enter to finish state 3...")
            # put in picture logic
        if STATE == 4:
            print("Robot 2 Take Picture------------------DO NOTHING")
            # Do Nothing
        if STATE == 5:
            input("Clear side------------------Press Enter to finish state 5...")
            # put in clearing logic
        if STATE == 6:
            #print("Finished---------------------GO HOME")
            input("Press Enter when home")
       
        if STATE > MAX_STATE:
            print("Master done.")
            client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "stop": True}))
            FINISHED = True
            client.disconnect()
            return
        print(f"Master -> State {STATE}")
        time.sleep(0.5)
        client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "state": STATE}))
    elif s < STATE:
        # Slave behind — remind it of current state
        client.publish(TOPIC, json.dumps({"sender": CLIENT_ID, "state": STATE}))

client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_forever()

print("finish routine and go home...................")