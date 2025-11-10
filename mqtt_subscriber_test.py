import paho.mqtt.client as mqtt
client_id = 'uicda'

# Callback when the client connects
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker!")
        client.subscribe("topic/uicdatest1")
    else:
        print("Failed to connect, return code:", rc)

# Callback when a message is received
def on_message(client, userdata, msg):
    print(f"Received on {msg.topic}: {msg.payload.decode()}")


# Create client instance
client = mqtt.Client()

# Assign event callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to local Mosquitto broker
client.connect("10.8.4.11", 1883, 60)

# Keep the connection alive and listening
client.loop_forever()