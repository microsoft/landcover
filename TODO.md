TODO microsoft/landcover repo
- Debug the saving/restoring models workflow
- Figure out how the remote calls work when using other ServerModels besides the Keras one
- Clean up unused branches
- There is an issue in DataLoader.warp_data_to_3857 where we assume that the destination resolution is the same as the input resolution. I think this is "fine" if both src and dst CRS are in meters (dst CRS will always be meters as it is EPSG:3857), however the src CRS is not garunteed to be this. I need to figure out a way to convert degree resolution to meters here if necessary.
- Resolution of a dataset should be specificed in datasets.json, and the front-end should use this to clip predictions and corrections to the nearest resolution unit. This should be handled in utils.js - getPolyAround().
- ServerModelsKeras with the demo models are not using the gpu