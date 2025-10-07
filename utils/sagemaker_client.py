import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='YOUR_REGION')

def invoke_sagemaker_endpoint(endpoint_name, payload):
    """
    Sends payload to SageMaker endpoint and returns prediction
    """
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    result = response['Body'].read().decode()
    return json.loads(result)
