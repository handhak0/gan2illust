import hashlib
import hmac
import base64
import time
import requests
import json


class ChatbotMessageSender():
    # chatbot api gateway url
    ep_path = 'https://e3146fb74e884b259f54fa080abdbbd9.apigw.ntruss.com/custom/v1/5218/7705c670ca83b21c80363195d1bb99601672122ed287f7586b8486e90e09ffa3'
    # chatbot custom secret key
    secret_key = 'S1pvbHZQWXNOdkRqem1QbG5XamJia0FrUGFDdnBxZVY='

    def __init__(self, question):
        self.question = question

    def req_message_send(self):
        timestamp = self.get_timestamp()
        request_body = {
            'version': 'v2',
            'userId': 'U47b00b58c90f8e47428af8b7bddcda3d1111111',
            'timestamp': timestamp,
            'bubbles': [
                {
                    'type': 'text',
                    'data': {
                        'description': self.question
                    }
                }
            ],
            'event': 'send'
        }

        ## Request body
        encode_request_body = json.dumps(request_body).encode('UTF-8')

        ## make signature
        signature = self.make_signature(self.secret_key, encode_request_body)

        ## headers
        custom_headers = {
            'Content-Type': 'application/json;UTF-8',
            'X-NCP-CHATBOT_SIGNATURE': signature
        }

        print("## Timestamp : ", timestamp)
        print("## Signature : ", signature)
        print("## headers ", custom_headers)
        print("## Request Body : ", encode_request_body)

        ## POST Request
        response = requests.post(headers=custom_headers, url=self.ep_path, data=encode_request_body)

        return response

    @staticmethod
    def get_timestamp():
        timestamp = int(time.time() * 1000)
        return timestamp

    @staticmethod
    def make_signature(secret_key, request_body):
        secret_key_bytes = bytes(secret_key, 'UTF-8')

        signing_key = base64.b64encode(hmac.new(secret_key_bytes, request_body, digestmod=hashlib.sha256).digest())

        return signing_key