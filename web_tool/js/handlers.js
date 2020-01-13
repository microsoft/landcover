var addInferenceMouseHandlers = function(){
    //----------------------------------------------------------------------
    // Setup map selection handlers
    //----------------------------------------------------------------------
    gMap.addEventListener('mousemove', function(e){
        // Choose style
        var curSelPoly = null;
        if(!gShiftKeyDown){
            curSelPoly = getPolyAround(e.latlng, CORRECTION_WINDOW_SIZE);
        }else{
            curSelPoly = getPolyAround(e.latlng, INFERENCE_WINDOW_SIZE);
        }
        
        if(gSelectionBox === null){
            gSelectionBox = L.polygon(curSelPoly, {
                color: "#000000",
                fillColor: "#ffffff",
                weight: 2
            });
            gSelectionBox.addTo(gMap);
        }else{
            if(!gAnimating){
                gSelectionBox.setStyle({
                    color: "#000000",
                    fillColor: "#ffffff",
                    weight: 2
                });
            }
            gSelectionBox.setLatLngs(curSelPoly);
        }
    });
    
    gMap.addEventListener('click', function(e){
        
        var curSelPoly = null;
        if(gShiftKeyDown){
            // Run the inference path
            curSelPoly = getPolyAround(e.latlng, INFERENCE_WINDOW_SIZE);
            if(gCurrentSelection === null){ // This condition creates the red selection box on the first click
                gCurrentSelection = L.polygon(curSelPoly, {
                    color: "#ff0000",
                    fillColor: "#ffffff",
                    weight: 2
                });
                gCurrentSelection.addTo(gMap);
            }else{
                gCurrentSelection.setLatLngs(curSelPoly);
            }
    
            requestPatches(curSelPoly);
        }else{
            // Run the add sample path
            if(gCurrentSelection !== null){
                if(isPointInsidePolygon(e.latlng, gCurrentSelection)){
                    if(gCurrentBasemapLayerName == DATASETS[gCurrentDataset]["metadata"]["imageryName"]){
                        curSelPoly = getPolyAround(e.latlng, CORRECTION_WINDOW_SIZE);
                        var idx = gCurrentPatches.length-1;
                        doSendCorrection(curSelPoly, idx);
                        
                        var rect = L.rectangle(
                            [curSelPoly[0], curSelPoly[2]],
                            {
                                color: CLASSES[gSelectedClassIdx]["color"],
                                weight: 1,
                                opacity: 1
                                //pane: "labels"
                            }
                        ).addTo(gMap);
                        gUserPointList.push([rect, gSelectedClassIdx]);
    
                        gMap.dragging.disable();
                        gNumClicks += 1
                        window.setTimeout(function(){
                            gNumClicks -= 1;
                            if(gNumClicks == 0){
                                gMap.dragging.enable();
                            }
                        }, 700);
                    }else{
                        notifyFailMessage("Please add corrections using the '"+DATASETS[gCurrentDataset]["metadata"]["imageryName"]+"' imagery layer.")
                    }
                }else{
                    console.debug("Click not in selection");
                }
            }
        }
    });

};


var addDrawControlHandlers = function(){
    gMap.on("draw:created", function (e) {
        var layer = e.layer;
        var type = e.layerType;
        
        if (type === 'polygon') {
            L.Util.setOptions(layer, {pane: "customPolygons"});
            if(gCurrentCustomPolygon !== null){
                gCurrentCustomPolygon.remove();
            }
            layer.addTo(gCustomDrawnItems);
            gCurrentCustomPolygon = layer;
        }
    });
    
    gMap.on("draw:deleted", function(e){
        var layer = e.layer;
        var type = e.layerType;
    
        if (type === "draw:deleted"){
            gCurrentCustomPolygon = null;
        }
    });
};
