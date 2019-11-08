# TODO: can we combine login.py and login_helper.py?
import jwt
import login_config as cfg

from azure.cosmosdb.table.tableservice import TableService # TODO: find out if these objects can be created everytime a connection is needed, or if they should only be created once

def found_user(emailaddress):
    table_service = TableService(account_name=cfg.ACCOUNT_NAME, account_key=cfg.ACCOUNT_KEY)

    user = table_service.query_entities( # TODO: find out if this is the best way to check for existence of a key
        'landcoverdemousers', 
        filter="PartitionKey eq 'user' and RowKey eq '{}'".format(emailaddress), 
        num_results=1
    )
    
    if len(user.items) > 0:
        return True
    return False

def check_user_access(graphdata, jwt_token):
    if found_user(graphdata['userPrincipalName']):
        return graphdata['userPrincipalName']
    elif found_user(graphdata['mail']):
        return graphdata['mail']
    else:
        jwt_token_decoded = jwt.decode(jwt_token, verify=False)
        
        if 'email' in jwt_token_decoded:
            email_address = str(jwt_token_decoded["email"]).strip("[]'")
            if found_user(email_address):
                return email_address
        if 'verified_primary_email' in jwt_token_decoded:
            email_address = str(jwt_token_decoded["verified_primary_email"]).strip("[]'")
            if found_user(email_address):
                return email_address
        
    return None

def get_token_from_querystring(query_string):
    for item in query_string:
        if 'id_token' in item:
            jwt_token = item.replace('id_token=','')
            return jwt_token
    return None