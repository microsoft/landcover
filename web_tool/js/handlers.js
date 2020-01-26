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

var addRetrainKeyHandler = function(){
    $(document).keydown(function(e) {
        if(document.activeElement == document.body){
            if((e.which == 114 || e.which == 82) && !gCtrlKeyDown){ // "r" - retrain
                doRetrain();
            }
        }
    });
};

var addOpacityKeyHandlers = function(opacitySlider){
    $(document).keydown(function(e) {
        if(document.activeElement == document.body){ // only register if we are in document.body so that we don't fire events when typing in text boxes
            if(e.which == 97 || e.which == 65) { // "a" - set invisible
                gVisible = false;
                gMap.getPane('labels').style.opacity = 0.0;
                opacitySlider.slider.value = 0;
                opacitySlider._updateValue();
            } else if(e.which == 115 || e.which == 83) { // "s" - toggle between visibile and invisible
                if(gVisible){
                    gVisible = false;
                    gMap.getPane('labels').style.opacity = 0.0;
                    opacitySlider.slider.value = 0;
                    opacitySlider._updateValue();
                }else{
                    gVisible = true;
                    gMap.getPane('labels').style.opacity = 1.0;
                    opacitySlider.slider.value = 100;
                    opacitySlider._updateValue();
                }
            } else if(e.which == 100 || e.which == 68) { // "d" - set visible
                gVisible = true;
                gMap.getPane('labels').style.opacity = 1.0;
                opacitySlider.slider.value = 100
                opacitySlider._updateValue();
            }
        }
    });
};

var addGUIKeyHandlers = function(){
    $(document).keydown(function(e){
        gShiftKeyDown = e.shiftKey;
        gCtrlKeyDown = e.ctrlKey;
    });
    $(document).keyup(function(e){
        gShiftKeyDown = e.shiftKey;
        gCtrlKeyDown = e.ctrlKey;
    });

    document.getElementById("map").onselectstart = function(){ return false;} // remove the default behavior of the shift key. TODO: should this be on "document" or "body" instead of "map"?
    gMap.on('contextmenu',function(e){});
    gMap.on('dblclick',function(e){});
    gMap.doubleClickZoom.disable();
    gMap.boxZoom.disable();
};