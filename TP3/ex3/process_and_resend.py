import json
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('duport', bootstrap_servers=['nowledgeable.com:9092'])

producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])

print("Process and resend script started, waiting for messages...")

for message in consumer:
    try:
        json_string = message.value.decode('utf-8')
        data_dict = json.loads(json_string)
        
        data_array = data_dict['data']
        np_array = np.array(data_array)
        total_sum = np.sum(np_array)
        
        result_message = {
            "sum": int(total_sum)  # Convert to int for JSON serialization
        }

        json_result = json.dumps(result_message)
        encoded_result = json_result.encode('utf-8')
        
        producer.send('processed', encoded_result)
        
        print(f"Processed data: {data_array}, Sum: {total_sum}, Sent to 'processed' topic")
        
    except Exception as e:
        print(f"Error processing message: {e}")

producer.close()