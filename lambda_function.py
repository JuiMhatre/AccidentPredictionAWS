# import json

# def lambda_handler(event, context):
#     # TODO implement
#     return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }


import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
                                       
    result = json.loads(response['Body'].read().decode())


    pred = float(result)
    predicted_label = f"Prediction of {data} is : "


    if pred == 0:
        predicted_label += 'You are good to go!'
    else:
        predicted_label += 'Beep Beep!... Accident can happen'
    
    print("Predicted Label : ", predicted_label)
    
    return predicted_label
