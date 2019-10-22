import os
import uuid
import urllib
import bottle
import shutil
import requests
import login_config as cfg

from login_helper import * # TODO: don't import *
from datetime import datetime
from bottle import request, redirect, template

from log import LOGGER

session_base_path = './data/session'
session_folder = session_base_path + "/" + datetime.now().strftime('%Y-%m-%d')

def manage_session_folders():
    if not os.path.exists(session_base_path):
        os.makedirs(session_base_path)

    if not os.path.exists(session_folder):
        shutil.rmtree(session_base_path)
        os.makedirs(session_folder)

def authenticated(func):
    def wrapped(*args, **kwargs):
        try:
            name = request.session['logged_in']
            return func(*args, **kwargs)
        except:
            redirect('/login')

    return wrapped

def load_authorized():
    return template('redirecting.tpl')

def load_error():
    return template('error.tpl')

def not_authorized():
    return template('not_authorized.tpl')

def do_login():
    auth_state = str(uuid.uuid4())

    nonce = uuid.uuid4().hex + uuid.uuid1().hex

    url = cfg.AUTHORITY_URL + '/oauth2/v2.0/authorize?response_type=id_token+token&'

    params = urllib.parse.urlencode({'client_id': cfg.CLIENT_ID,
                                     'redirect_uri': cfg.REDIRECT_URI,
                                     'state': auth_state,
                                     'nonce': nonce,
                                     'prompt': 'select_account',
                                     'scope': 'user.read openid profile'})

    return bottle.redirect(url + params)

def do_logout():
    request.session['logged_in'] = None
    request.session.delete()
   
    return template('landing_page.tpl')

def get_accesstoken():
    access_token = request.forms.get("token").split("#")[1].split("&")[0]
    access_token = access_token.replace("access_token=", '')

    query_string =  request.forms.get("token").split("#")[1].split("&")
    jwt_token = get_token_from_querystring(query_string)

    endpoint = cfg.RESOURCE + cfg.API_VERSION + '/me'
    http_headers = {'Authorization': 'Bearer {}'.format(access_token)}
    graphdata = requests.get(endpoint, headers=http_headers, stream=False).json()
        
    if graphdata.get('userPrincipalName'):
        if check_user_access(graphdata, jwt_token):
            request.session['logged_in'] = "yes"
            redirect("/")
        else:
            if(cfg.LOG_TOKEN):
                LOGGER.debug("Not authorized")
                LOGGER.debug(graphdata)
                LOGGER.debug("access_token="+access_token)
                LOGGER.debug("JWToken=" + jwt_token)
            
            redirect("/notAuthorized")
    else:
        LOGGER.debug("Error- No graph data")
        if(access_token):
            LOGGER.debug("Accesstoken=" + access_token)
        if(jwt_token):
            LOGGER.debug("JWToken=" + jwt_token)

        redirect("/error")
