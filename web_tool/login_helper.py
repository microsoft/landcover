import ast
import jwt #(pip install pyjwt)
import login_config as cfg
import base64

from azure.cosmosdb.table.tableservice import TableService

table_service = TableService(account_name=cfg.ACCOUNT_NAME, account_key=cfg.ACCOUNT_KEY)

def found_user(emailaddress):
    user = table_service.query_entities('landcoverdemousers', 
           filter="PartitionKey eq 'user' and RowKey eq '{}'".format(emailaddress), 
           num_results=1)
    
    if len(user.items) > 0:
        return True
    return False

def check_user_access(graphdata, jwt_token):
    if found_user(graphdata['userPrincipalName']):
        return True
    elif found_user(graphdata['mail']):
        return True
    else:
        jwt_token_decoded = jwt.decode(jwt_token, verify=False)
        
        if 'email' in jwt_token_decoded:
            email_address = str(jwt_token_decoded["email"]).strip("[]'")
            if found_user(email_address):
                return True
        if 'verified_primary_email' in jwt_token_decoded:
            email_address = str(jwt_token_decoded["verified_primary_email"]).strip("[]'")
            if found_user(email_address):
                return True
        
    return False

def get_token_from_querystring(query_string):
    for item in query_string:
        if 'id_token' in item:
            jwt_token = item.replace('id_token=','')
            return jwt_token
    return None

def decode_token(jwt_token):
    if '.' in jwt_token:
        base64Url = jwt_token.split('.')[1]
    else:
        base64Url = jwt_token

    base64_str = base64Url.replace(r'/-/g', '+').replace(r'/_/g', '/')
    decoded_str = base64.b64decode(base64_str+"==")
    try:
        decoded_str =  str(decoded_str, 'utf-8')
    except:
        decoded_str = str(decoded_str, 'latin-1')
    decoded_str = decoded_str.replace(chr(0), '');
    decoded = ast.literal_eval(decoded_str)
    return decoded

