#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Caleb Robinson <calebrob6@gmail.com>
#
'''Script for creating the shapefile used in `create_spatial_index.py`
'''
import sys, os, time
import numpy as np
import rasterio
import fiona.transform
from multiprocessing import Process, Queue
import queue
import json

BASE_URL = "https://naipblobs.blob.core.windows.net/naip/"
NUM_PROCESSES = 12


def get_geom_from_bounds(bounds):
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    return {
       "type": "Polygon",
       "coordinates": [[[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]]
    }


def do_work(work_queue, output_queue, process_idx):
    tic = float(time.time())
    while not work_queue.empty():
        try:
            fn = work_queue.get(block=True, timeout=5)
            with rasterio.open(BASE_URL + fn) as f:
                geom = fiona.transform.transform_geom(f.crs.to_string(), "EPSG:4326", get_geom_from_bounds(f.bounds))
                output_queue.put((geom, fn))
        except queue.Empty:
            print("Empty queue, doing nothing")
        except Exception as e:
            print("ERROR:")
            print(e)
    output_queue.put(None)
            
            
def output_monitor(output_queue, output_fn, num_workers, num_events):
    with open(output_fn, "w") as f:
        pass
    
    tic = float(time.time())
    num_workers_finished = 0
    num_events_processed = 0
    while num_workers_finished < num_workers:
        
        output = output_queue.get()

        if output is not None:
            geom, fn = output
            
            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "fn": fn,
                }
            }
            
            with open(output_fn, "a") as f:
                f.write(json.dumps(feature))
                f.write("\n")
          
            num_events_processed += 1
            if num_events_processed % 100 == 0:
                print("(%d/%d)\t%0.4f%%\t%0.4f seconds" % (num_events_processed, num_events, num_events_processed/num_events*100, time.time()-tic))
                tic = float(time.time())
        else:
            num_workers_finished += 1


def main():

    if len(sys.argv) < 3:
        print("Usage: ./create_shapefile_from_files.py PATH_TO_FILELIST PATH_TO_OUTPUT_SHAPEFILE")
        return

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    assert os.path.exists(input_fn)
    assert not os.path.exists(output_fn)


    #-----------------------------------
    with open(input_fn, "r") as f:
        fns = f.read().strip().split("\n")
    print("Found %d files" % (len(fns)))


    #-----------------------------------
    work_queue = Queue()
    output_queue = Queue()

    num_events = 0
    for fn in fns:
        work_queue.put(fn)
        num_events += 1


    #-----------------------------------
    processes = []
    for process_idx in range(NUM_PROCESSES):
        p = Process(target=do_work, args=(work_queue, output_queue, process_idx))
        processes.append(p)
        p.start()

    p = Process(target=output_monitor, args=(output_queue, output_fn + ".txt", NUM_PROCESSES, num_events))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()


    #-----------------------------------
    with open(output_fn + ".txt", "r") as f:
        features = f.read().strip().split("\n")
        
    output = {
        "type": "FeatureCollection",
        "name": "output",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": [
            json.loads(feature)
            for feature in features
        ]
    }

    with open(output_fn, "w") as f:
        f.write(json.dumps(output))

    os.remove(output_fn + ".txt")


if __name__ == "__main__":
    main()
