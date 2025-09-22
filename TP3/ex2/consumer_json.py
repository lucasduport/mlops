import json
import numpy as np
from kafka import KafkaConsumer

# Create a Kafka consumer for the topic
consumer = KafkaConsumer('duport', bootstrap_servers=['nowledgeable.com:9092'])

print("Consumer started, waiting for messages...")

for message in consumer:
    # Decode the received message from bytes to string
    json_string = message.value.decode('utf-8')
    
    # Parse the JSON string to a dictionary
    data_dict = json.loads(json_string)
    
    # Extract the data array
    data_array = data_dict['data']
    
    # Convert to numpy array
    np_array = np.array(data_array)
    
    # Calculate the sum of all values
    total_sum = np.sum(np_array)
    
    # Display the result
    print(f"Received data: {data_array}")
    print(f"Numpy array: {np_array}")
    print(f"Sum of values: {total_sum}")
