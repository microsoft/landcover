import os
import uuid
import urllib
import bottle
import shutil
import requests
import login_config as cfg

from login_helper import * # TODO: don't import *; can we combine login.py and login_helper.py?
from Session import Session
from datetime import datetime
import bottle

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
    '''Based on suggestion from https://stackoverflow.com/questions/11698473/bottle-hooks-with-beaker-session-middleware-and-checking-logins
    '''
    def wrapped(*args, **kwargs):
        if 'logged_in' in bottle.request.session:
            return func(*args, **kwargs)
        else:
            LOGGER.info("User not logged in")
            bottle.abort(401, "Sorry, access denied.")
            #bottle.redirect('/login')
    return wrapped

def load_authorized():
    return bottle.template('redirecting.tpl')

def load_error():
    return bottle.template('error.tpl')

def not_authorized():
    return bottle.template('not_authorized.tpl')

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
    bottle.request.session.delete()   
    return bottle.template('landing_page.tpl')

def get_accesstoken(SESSION_MAP):
    if "logged_in" in bottle.request.session:
        bottle.redirect("/")
    else:
        access_token = bottle.request.forms.get("token").split("#")[1].split("&")[0]
        access_token = access_token.replace("access_token=", '')

        query_string =  bottle.request.forms.get("token").split("#")[1].split("&")
        jwt_token = get_token_from_querystring(query_string)

        endpoint = cfg.RESOURCE + cfg.API_VERSION + '/me'
        http_headers = {'Authorization': 'Bearer {}'.format(access_token)}
        graphdata = requests.get(endpoint, headers=http_headers, stream=False).json()
            
        if graphdata.get('userPrincipalName'):
            name = check_user_access(graphdata, jwt_token) 
            if name is not None:
                bottle.request.session['logged_in'] = True
                bottle.request.session['name'] = str(name)
                SESSION_MAP[bottle.request.session.id] = Session(bottle.request.session.id)
                bottle.redirect("/")
            else:
                if(cfg.LOG_TOKEN):
                    LOGGER.debug("Not authorized")
                    LOGGER.debug(graphdata)
                    LOGGER.debug("access_token="+access_token)
                    LOGGER.debug("JWToken=" + jwt_token)
                
                bottle.redirect("/notAuthorized")
        else:
            LOGGER.debug("Error- No graph data")
            if(access_token):
                LOGGER.debug("Accesstoken=" + access_token)
            if(jwt_token):
                LOGGER.debug("JWToken=" + jwt_token)

            bottle.redirect("/error")
