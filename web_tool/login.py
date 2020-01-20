#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1101
import os
import datetime
import uuid
import urllib
import shutil
import requests

import login_config
from login_helper import check_user_access, get_token_from_querystring
from log import LOGGER

import bottle

SESSION_BASE_PATH = './data/session'
SESSION_FOLDER = SESSION_BASE_PATH + "/" + datetime.datetime.now().strftime('%Y-%m-%d')


def manage_session_folders():
    if not os.path.exists(SESSION_BASE_PATH):
        os.makedirs(SESSION_BASE_PATH)

    if not os.path.exists(SESSION_FOLDER):
        shutil.rmtree(SESSION_BASE_PATH)
        os.makedirs(SESSION_FOLDER)


def authenticated(LOCAL_MANAGER):
    def authenticated_inner(func):
        '''Based on suggestion from https://stackoverflow.com/questions/11698473/bottle-hooks-with-beaker-session-middleware-and-checking-logins
        '''
        def wrapped(*args, **kwargs):
            if ('logged_in' in bottle.request.session) or LOCAL_MANAGER.run_local:
                return func(*args, **kwargs)
            else:
                LOGGER.info("User not logged in")
                bottle.abort(401, "Sorry, access denied.")
                #bottle.redirect('/login')
        return wrapped
    return authenticated_inner

def load_authorized():
    return bottle.template('redirecting.tpl')

def load_error():
    return bottle.template('error.tpl')

def not_authorized():
    return bottle.template('not_authorized.tpl')

def do_login():
    if "logged_in" in bottle.request.session:
        bottle.redirect("/")
    else:
        auth_state = str(uuid.uuid4())

        nonce = uuid.uuid4().hex + uuid.uuid1().hex

        url = login_config.AUTHORITY_URL + '/oauth2/v2.0/authorize?response_type=id_token+token&'

        params = urllib.parse.urlencode({
            'client_id': login_config.CLIENT_ID,
            'redirect_uri': login_config.REDIRECT_URI,
            'state': auth_state,
            'nonce': nonce,
            'prompt': 'select_account',
            'scope': 'user.read openid profile'
        })

        return bottle.redirect(url + params)

def do_logout():
    bottle.request.session.delete()   
    return bottle.template('front_page.tpl')

def get_accesstoken(SESSION_MAP, session_factory):
    if "logged_in" in bottle.request.session:
        bottle.redirect("/")
    else:
        access_token = bottle.request.forms.get("token").split("#")[1].split("&")[0]
        access_token = access_token.replace("access_token=", '')

        query_string =  bottle.request.forms.get("token").split("#")[1].split("&")
        jwt_token = get_token_from_querystring(query_string)

        endpoint = login_config.RESOURCE + login_config.API_VERSION + '/me'
        http_headers = {'Authorization': 'Bearer {}'.format(access_token)}
        graphdata = requests.get(endpoint, headers=http_headers, stream=False).json()
            
        if graphdata.get('userPrincipalName'):
            name = check_user_access(graphdata, jwt_token) 
            if name is not None:
                bottle.request.session['logged_in'] = True
                bottle.request.session['name'] = str(name)
                SESSION_MAP[bottle.request.session.id] = session_factory.get_session(bottle.request.session.id)
                SESSION_MAP[bottle.request.session.id].spawn_worker()
                bottle.redirect("/")
            else:
                if(login_config.LOG_TOKEN):
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
