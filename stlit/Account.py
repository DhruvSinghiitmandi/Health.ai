from google.oauth2 import service_account
from googleapiclient.discovery import build
import time
from datetime import datetime
from googleapiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os.path
import json
import streamlit as st
import os
import streamlit.components.v1 as com
import subprocess
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = "1"


# credentials from the Google Developers Console
CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

# Check https://developers.google.com/fit/rest/v1/reference/users/dataSources/datasets/get
# for all available scopes
OAUTH_SCOPE = 'https://www.googleapis.com/auth/fitness.activity.read'

# DATA SOURCE
DATA_SOURCE = "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
# The ID is formatted like: "startTime-endTime" where startTime and endTime are
# 64 bit integers (epoch time with nanoseconds).
TODAY = datetime.today().date()
NOW = datetime.today()
START = int(time.mktime(TODAY.timetuple())*1000000000)
END = int(time.mktime(NOW.timetuple())*1000000000)
DATA_SET = "%s-%s" % (START, END)

creds = None

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

CLIENT_FILE = 'creds.json'
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read', 'https://www.googleapis.com/auth/fitness.blood_glucose.read', 'https://www.googleapis.com/auth/fitness.blood_pressure.read', 'https://www.googleapis.com/auth/fitness.body.read', 'https://www.googleapis.com/auth/fitness.body_temperature.read', 'https://www.googleapis.com/auth/fitness.heart_rate.read', 'https://www.googleapis.com/auth/fitness.location.read', 'https://www.googleapis.com/auth/fitness.nutrition.read', 'https://www.googleapis.com/auth/fitness.oxygen_saturation.read', 'https://www.googleapis.com/auth/fitness.reproductive_health.read', 'https://www.googleapis.com/auth/fitness.sleep.read', 'https://www.googleapis.com/auth/userinfo.profile']

REDIRECT_URI = 'http://localhost:8080'  # Use a fixed port
TOKEN_FILE = 'token.json'

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as token:
            creds_data = json.load(token)
            creds = Credentials.from_authorized_user_info(creds_data, SCOPES)
    return creds

creds = get_credentials()

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        st.markdown(f'<img src="https://i.imgur.com/ngr2HSn.png" width="200">',
                    unsafe_allow_html=True) 
        st.markdown('##')
        st.write('Welcome to Health.AI ! Please sign in with your Google Account to continue.')
        if st.button('Sign In with Google'):
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_FILE, SCOPES, redirect_uri=REDIRECT_URI)
            creds = flow.run_local_server(port=8080)
            print(creds)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                st.rerun()
else:
    if st.button('Continue to Chat'):
        st.switch_page('pages/Home.py')

    if st.button('Sign Out'):
        subprocess.Popen("rm ./token.json",shell=True)
        time.sleep(0.5)
        st.rerun()
        




