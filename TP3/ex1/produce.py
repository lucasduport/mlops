from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])

producer.send('exo1', b'coucou Lucas')

producer.close()
print("Producer finished.")