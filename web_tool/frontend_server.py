#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''Easy HTTP server for serving the front end
'''
import sys
import os
import bottle
import argparse


ROOT_DIR = "web_tool"


@bottle.get("/")
def root_app():
    return bottle.static_file("index.html", root="./" + ROOT_DIR + "/")

@bottle.get("/favicon.ico")
def favicon():
    return

@bottle.get("/<filepath:re:.*>")
def everything_else(filepath):
    return bottle.static_file(filepath, root="./" + ROOT_DIR + "/")

def main():
    parser = argparse.ArgumentParser(description="Frontend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4040)

    args = parser.parse_args(sys.argv[1:])

    # TODO: Check for environment variables that should override the host and port
    
    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "server": "tornado",
        "reloader": False
    }
    bottle.run(**bottle_server_kwargs)
    return

if __name__ == '__main__':
    main()
