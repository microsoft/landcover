#!/bin/bash
cd /lf
./Microsoft.LocalForwarder.ConsoleHost noninteractive &
python /app/fuse/blob_mounter.py
/usr/bin/supervisord