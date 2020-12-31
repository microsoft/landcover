//-----------------------------------------------------------------
// Retrain backend server
//-----------------------------------------------------------------
var doRetrain = function(){
    var request = {
        "type": "retrain",
        "retrainArgs": gRetrainArgs,
    };
    $.ajax({
        type: "POST",
        url: gBackendURL + "retrainModel",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            if(data["success"]){
                notifySuccess(data, textStatus, jqXHR, 5000);
                gRetrainCounts += 1;

                $("#label-retrains").html(gRetrainCounts);
                $(".classCounts").html("0")
                for(c in CLASSES){
                    CLASSES[c]["count"] = 0;
                }

                var t = gCurrentSelection._latlngs[0];
                var curSelPoly = [
                    [t[0]["lat"], t[0]["lng"]],
                    [t[1]["lat"], t[1]["lng"]],
                    [t[2]["lat"], t[2]["lng"]],
                    [t[3]["lat"], t[3]["lng"]]
                ];
                requestPatches(curSelPoly);
            }
        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Reset backend server state
//-----------------------------------------------------------------
var doReset = function(){
    var request = {
        "type": "reset",
    };
    $.ajax({
        type: "POST",
        url: gBackendURL + "resetModel",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            if(data["success"]){
                notifySuccess(data, textStatus, jqXHR);
                
                $("#label-retrains").html("0");
                $(".classCounts").html("0")
                for(c in CLASSES){
                    CLASSES[c]["count"] = 0;
                }
            }
            // TODO: Reset the class list
        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Download tile
//-----------------------------------------------------------------
var doDownloadTile = function(){

    var polygon = null;
    if(gCurrentZone !== null){
        polygon = gCurrentZone;
    }else{
        new Noty({
            type: "info",
            text: "You must select a zone or draw a polygon to download data",
            layout: 'topCenter',
            timeout: 3000,
            theme: 'metroui'
        }).show();
        return
    }

    var request = {
        "type": "download",
        "dataset": gCurrentDataset,
        "polygon": polygon.toGeoJSON(),
        "classes": CLASSES,
        "zoneLayerName": null,
        "modelIdx": parseInt(gActiveImgIdx),
    };

    var outputLayer = L.imageOverlay("", polygon.getBounds(), {pane: "labels"}).addTo(gMap);
    $.ajax({
        type: "POST",
        url: gBackendURL + "predTile",
        data: JSON.stringify(request),
        timeout: 0,
        success: function(data, textStatus, jqXHR){
            new Noty({
                type: "success",
                text: "Tile download ready!",
                layout: 'topCenter',
                timeout: 5000,
                theme: 'metroui'
            }).show();
            var pngURL = window.location.origin + "/" + data["downloadPNG"];
            var tiffURL = window.location.origin + "/" + data["downloadTIFF"];
            var statisticsURL = window.location.origin + "/" + data["downloadStatistics"];
            $("#lblPNG").html("<a href='"+pngURL+"' target='_blank'>Download PNG</a>");
            $("#lblTIFF").html("<a href='"+tiffURL+"' target='_blank'>Download TIFF</a>");
            $("#lblStatistics").html("<a href='"+statisticsURL+"' target='_blank'>Download Class Statistics</a>");

            outputLayer.setUrl(pngURL);

        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });

    new Noty({
        type: "success",
        text: "Sent tile download request, please wait for 2-3 minutes. When the request is complete the download links will appear underneath the 'Download' button.",
        layout: 'topCenter',
        timeout: 10000,
        theme: 'metroui'
    }).show();
};

//-----------------------------------------------------------------
// Submit new training example
//-----------------------------------------------------------------
var doSendCorrection = function(point, idx){
    var pointProjected = L.CRS.EPSG3857.project(point);
    
    var request = {
        "type": "correction",
        "dataset": gCurrentDataset,
        "point": {
            "x": Math.round(pointProjected.x),
            "y": Math.round(pointProjected.y),
            "crs": "epsg:3857"
        },
        "classes": CLASSES,
        "value" : gSelectedClassIdx,
        "modelIdx": parseInt(gActiveImgIdx),
    };

    var value = gSelectedClassIdx;
    $.ajax({
        type: "POST",
        url: gBackendURL + "recordCorrection",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            CLASSES[value]["count"] += 1;
            renderClassCount(CLASSES[value]["name"], CLASSES[value]["count"]);
            animateSuccessfulCorrection(10, 80);
        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Submit an undo request
//-----------------------------------------------------------------
var doUndo = function(){

    var request = {
        "type": "undo",
        "dataset": gCurrentDataset,
    };

    if(!gUndoInProgress){
        $.ajax({
            type: "POST",
            url: gBackendURL + "doUndo",
            data: JSON.stringify(request),
            success: function(data, textStatus, jqXHR){
                
                for(var i=0;i<data["count"];i++){
                    var removedPoint = gUserPointList.pop();
                    gMap.removeLayer(removedPoint[0]);
                    
                    var labelIdx = removedPoint[1];
                    CLASSES[labelIdx]["count"] -= 1;
                    renderClassCount(CLASSES[labelIdx]["name"], CLASSES[labelIdx]["count"]);
                }

                // alert success
                new Noty({
                    type: "success",
                    text: "Successful undo! Rewound " + data["count"] + " samples",
                    layout: 'topCenter',
                    timeout: 1000,
                    theme: 'metroui'
                }).show();
            }, 
            error: notifyFail,
            always: function(){
                gUndoInProgress = false;
            },
            dataType: "json",
            contentType: "application/json"
        });
    }else{
        new Noty({
            type: "error",
            text: "Please wait until current undo request finishes",
            layout: 'topCenter',
            timeout: 1000,
            theme: 'metroui'
        }).show();
    }
};

//-----------------------------------------------------------------
// Get predictions
//-----------------------------------------------------------------
var requestPatches = function(polygon){
    // Setup placeholders for the predictions from the current click to be saved to
    var idx = gCurrentPatches.length;
    gCurrentPatches.push({
        "naipImg": null,
        "imageLayer": L.imageOverlay("", L.polygon(polygon).getBounds(), {pane: "labels"}).addTo(gMap),
        "patches": [],
        "activeImgIdx": gActiveImgIdx
    });
    gCurrentPatches[idx]["patches"].push({
        "srcs": null
    });
    
    requestInputPatch(idx, polygon, gBackendURL);
    requestPatch(idx, polygon, 0, gBackendURL);

    // The following code is for connecting to multiple backends at once
    // for(var i=0; i<ENDPOINTS.length; i++){
    //     //console.debug("Running requestPatch on " + ENDPOINTS[i]["url"]);
    //     gCurrentPatches[idx]["patches"].push({
    //         "srcs": null
    //     });
    //     requestPatch(idx, polygon, i, gBackendURL); //TODO: this should be changed if we want to have a web tool that queries different backends
    // }
};

var requestPatch = function(idx, polygon, currentImgIdx, serviceURL){
    var topleft = L.latLng(polygon[0][0], polygon[0][1]);
    var topleftProjected = L.CRS.EPSG3857.project(topleft);
    var bottomright = L.latLng(polygon[2][0], polygon[2][1]);
    var bottomrightProjected = L.CRS.EPSG3857.project(bottomright);

    var request = {
        "type": "runInference",
        "dataset": gCurrentDataset,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "crs": "epsg:3857"
        },
        "classes": CLASSES,
    };
    
    $.ajax({
        type: "POST",
        url: serviceURL + "predPatch",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            let resp = data;

            var srcs = [{
                "soft": "data:image/png;base64," + resp.output_soft,
                "hard": "data:image/png;base64," + resp.output_hard,
            }];
            
            for(var i=0; i<1; i++){
                // Display the result on the map if we are the currently selected model
                let tSelection = gDisplayHard ? "hard" : "soft";
                if(i == gCurrentPatches[idx]["activeImgIdx"]){
                    gCurrentPatches[idx]["imageLayer"].setUrl(srcs[i][tSelection]);
                }
    
                // Save the resulting data in all cases
                gCurrentPatches[idx]["patches"][i]["srcs"] = srcs[i];
    
                // Update the right panel if we are the current "last item", we need to check for this because the order we send out requests to the API isn't necessarily the order they will come back
                if(idx == gCurrentPatches.length-1){
                    var img = $("#exampleImage_"+i);
                    img.attr("src", srcs[i][tSelection]);
                    img.attr("data-name", resp.model_name);                    
                    
                    if(i == gCurrentPatches[idx]["activeImgIdx"]){
                        img.addClass("active");
                    }
                }
            }

        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Get NAIP input
//-----------------------------------------------------------------
var requestInputPatch = function(idx, polygon, serviceURL){
    var topleft = L.latLng(polygon[0][0], polygon[0][1]);
    var topleftProjected = L.CRS.EPSG3857.project(topleft);
    var bottomright = L.latLng(polygon[2][0], polygon[2][1]);
    var bottomrightProjected = L.CRS.EPSG3857.project(bottomright);

    var request = {
        "type": "getInput",
        "dataset": gCurrentDataset,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "crs": "epsg:3857"
        },
    };

    $.ajax({
        type: "POST",
        url: serviceURL + "getInput",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            var resp = data;
            var inputImage = "data:image/png;base64," + resp.input_img;

            //gCurrentPatches[idx]["naipImg"] = naipImg
            
            // Update the right panel if we are the current "last item", we need to check for this because the order we send out requests to the API isn't necessarily the order they will come back
            if(idx == gCurrentPatches.length-1){
                $("#inputImage").attr("src", inputImage);
            }
        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Kill the current session on the backend and return to the landing page
//-----------------------------------------------------------------
var doKillSession = function () {
    $.ajax({
        type: "POST",
        url: window.location.origin + "/killSession",
        data: JSON.stringify({}),
        success: function (data, textStatus, jqXHR) {
            // TODO: Not sure if this is necessary. When we call `session.delete()` on the Beaker session on the server it will send an "expiration on the cookie requesting the browser to clear it" back to the browser. Will this be processed immediately, i.e. before we get here in the execution flow, or do we need to do this wait? See https://beaker.readthedocs.io/en/latest/sessions.html#removing-expired-old-sessions
            setTimeout(function(){window.location.href = "/"}, 1000); // Wait a bit so that the session cookie can be deleted
        },
        error: function (jqXHR, textStatus) {
            // TODO: Notify fail
        },
        dataType: "json",
        contentType: "application/json"
    });
};

//-----------------------------------------------------------------
// Create a checkpoint for the current model/dataset
//-----------------------------------------------------------------
var doCreateCheckpoint = function(){
    var name = prompt("Enter a unique name for checkpoint")

    var request = {
        "type": "createCheckpoint",
        "dataset": gCurrentDataset,
        "model": gCurrentModel,
        "checkpointName": name,
        "classes": CLASSES
    };

    $.ajax({
        type: "POST",
        url: window.location.origin + "/createCheckpoint",
        data: JSON.stringify(request),
        success: notifySuccess,
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });


}