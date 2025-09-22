import json
from kafka import KafkaProducer

# Create a Kafka producer
producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])

# Define the message data
message_data = {
    "data": [[1, 2], [3, 4]]
}

# Serialize the message to JSON string
json_message = json.dumps(message_data)

# Encode the JSON string to UTF-8 bytes
encoded_message = json_message.encode('utf-8')

# Send the message to the topic named after the last name
producer.send('duport', encoded_message)

# Close the producer
producer.close()
print("Producer finished sending JSON message.")
