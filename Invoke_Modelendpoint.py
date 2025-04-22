try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from io import BytesIO
import csv
import sys, os, base64, datetime, hashlib, hmac 
from chalice import Chalice
from chalice import NotFoundError, BadRequestError

import sys, os, base64, datetime, hashlib, hmac, json
app = Chalice(app_name='income-model')
app.debug = True

try:
    from urlparse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs

import boto3
sagemaker = boto3.client('sagemaker-runtime')

@app.route('/', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def handle_data():
    d = parse_qs(app.current_request.raw_body)
    # data to csv

    try:
        my_dict = {k:float(v[0]) for k, v in d.iteritems()}
        print(my_dict)
    except AttributeError:
        my_dict = {k:float(v[0]) for k, v in d.items()}
        print("There was an error")
        print(my_dict)
    f = StringIO()
    w = csv.DictWriter(f, my_dict.keys())
    #w.writeheader()
    w.writerow(my_dict)
    print("Ready to send request:")
    print(f.getvalue())
    res = sagemaker.invoke_endpoint(
                    EndpointName='income-model',
                    Body=f.getvalue(),
                    ContentType='text/csv'
                )
    print("Request complete")
    print(res)
    res_float = float(res['Body'].read())
    if res_float < 0.5:
        evaluation = f'This candidate is not eligible for government assistance.'
    else:
        evaluation = f'This candidate is eligible for government assistance.'
    return json.dumps(evaluation)
