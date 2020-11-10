import io  
import shutil  
import time
import os
from azure.storage.blob import BlockBlobService
import uuid
import azure_config as cfg

temp_path = "azure-temp-storage"

def delete_temp_folders():
    now = time.time()
    old = now - 1 * 24 * 60 * 60
    
    for root, dirs, files in os.walk(temp_path, topdown=False):
        for dir in dirs:
            if os.stat(os.path.join(os.getcwd(),temp_path, dir)).st_mtime < old: 
                delete_temp_folder(dir)

def delete_temp_folder(folder_name):
    try:
        path = os.path.join(os.getcwd(),temp_path, folder_name)
        shutil.rmtree(path)
    except Exception as e:
        print("Error occurred while deleting temp azure folder: " + str(e))

def get_blob(container_name, blob_name):
    
    delete_temp_folders()
    
    if(container_name == 'naip'):
        block_blob_service = BlockBlobService(account_name=cfg.NAIP_ACCOUNT_NAME, sas_token=cfg.NAIP_SAS_TOKEN)
    else:
        block_blob_service = BlockBlobService(account_name=cfg.MODELOUTPUT_ACCOUNT_NAME, 
                             account_key=cfg.MODELOUTPUT_ACCOUNT_KEY)

    file_name = blob_name.split("/").pop(-1)
    temp_sub_folder = str(uuid.uuid4())
    full_temp_folder = 'azure-temp-storage/' + temp_sub_folder
    file_path = full_temp_folder + "/" + file_name

    os.mkdir(full_temp_folder)

    if(file_name.endswith("mrf")):
        blob_parts = blob_name.split('/')
        year = blob_parts[2]
        state = blob_parts[4]
        state_resolution_year = blob_parts[5].replace('1m', '100cm')
        quadrangle = blob_parts[6]
        file_name = blob_parts[7].replace('.mrf', '.tif')
        file_path = full_temp_folder + "/" + file_name

        blob_name =  'v002/{}/{}/{}/{}/{}'.format(state, year, state_resolution_year, quadrangle, file_name)

        block_blob_service.get_blob_to_path(container_name, blob_name, file_path)

    elif(file_name.endswith("tif")):
        block_blob_service.get_blob_to_path(container_name, blob_name, file_path)


    return file_path, temp_sub_folder