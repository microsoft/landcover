//-----------------------------------------------------------------
// Retrain backend server
//-----------------------------------------------------------------
var doRetrain = function(){
    var request = {
        "type": "retrain",
        "dataset": DATASET,
        "experiment": EXP_NAME,
        "retrainArgs": retrainArgs
    };
    $.ajax({
        type: "POST",
        url: BACKEND_URL + "retrainModel",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            if(data["success"]){
                notifySuccess(data, textStatus, jqXHR, 5000);
                retrainCounts += 1;

                $("#label-retrains").html(retrainCounts);
                for( k in labelCounts){
                    labelCounts[k] = 0;
                    $("#label-counts-"+k).html("0");
                }

                var t = currentSelection._latlngs[0];
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
var doReset = function(notify=true){
    var request = {
        "type": "reset",
        "dataset": DATASET,
        "experiment": EXP_NAME
    };
    $.ajax({
        type: "POST",
        url: BACKEND_URL + "resetModel",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            if(data["success"] && notify){
                notifySuccess(data, textStatus, jqXHR);
                
                $("#label-retrains").html("0");

                for( k in labelCounts){
                    labelCounts[k] = 0;
                    $("#label-counts-"+k).html("0");
                }
            }
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
    var t = currentSelection._latlngs[0];
    var polygon = [
        [t[0]["lat"], t[0]["lng"]],
        [t[1]["lat"], t[1]["lng"]],
        [t[2]["lat"], t[2]["lng"]],
        [t[3]["lat"], t[3]["lng"]]
    ];

    var topleft = L.latLng(polygon[0][0], polygon[0][1]);
    var topleftProjected = L.CRS.EPSG3857.project(topleft);
    var bottomright = L.latLng(polygon[2][0], polygon[2][1]);
    var bottomrightProjected = L.CRS.EPSG3857.project(bottomright);

    var request = {
        "type": "download",
        "dataset": DATASET,
        "experiment": EXP_NAME,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "spatialReference": {
                "latestWkid": 3857
            }
        },
        "colors": colorList,
        "zoneLayerName": currentZoneLayerName
    };

    $.ajax({
        type: "POST",
        url: BACKEND_URL + "predTile",
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
            $("#lblPNG").html("<a href='"+pngURL+"' target='_blank'>Download PNG</a>");
            $("#lblTIFF").html("<a href='"+tiffURL+"' target='_blank'>Download TIFF</a>");
            
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
var doSendCorrection = function(polygon, idx){
    var topleft = L.latLng(polygon[0][0], polygon[0][1]);
    var topleftProjected = L.CRS.EPSG3857.project(topleft);
    var bottomright = L.latLng(polygon[2][0], polygon[2][1]);
    var bottomrightProjected = L.CRS.EPSG3857.project(bottomright);
    
    var request = {
        "type": "correction",
        "dataset": DATASET,
        "experiment": EXP_NAME,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "spatialReference": {
                "latestWkid": 3857
            }
        },
        "colors": colorList,
        "value" : selectedClassIdx
    };


    $.ajax({
        type: "POST",
        url: BACKEND_URL + "recordCorrection",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            console.debug("Successfully recorded correction");
            console.debug(data);

            labelName = findClassByIdx(data["value"])
            console.debug(labelName)
            //labelCounts[data["value"]] += data["count"];
            labelCounts[labelName] += 1;

            $("#label-counts-"+labelName).html(labelCounts[labelName]);
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
        "dataset": DATASET,
        "experiment": EXP_NAME,
    };

    if(!undoInProgress){
        $.ajax({
            type: "POST",
            url: BACKEND_URL + "doUndo",
            data: JSON.stringify(request),
            success: function(data, textStatus, jqXHR){
                
                // remove previously added point
                console.debug(data);

                for(var i=0;i<data["count"];i++){
                    var removedPoint = userPointList.pop();
                    map.removeLayer(removedPoint[0]);
                    var labelName = findClassByIdx(removedPoint[1]);

                    labelCounts[labelName] -= 1;
                    $("#label-counts-"+labelName).html(labelCounts[labelName]);
                }

                // 

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
                undoInProgress = false;
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
    currentPatches.push({
        "naipImg": null,
        "imageLayer": L.imageOverlay("", L.polygon(polygon).getBounds(), {pane: "labels"}).addTo(map),
        "patches": [],
        "activeImgIdx": activeImgIdx
    });
    var idx = currentPatches.length-1;
    
    requestInputPatch(idx, polygon, BACKEND_URL);

    currentPatches[idx]["patches"].push({
        "srcs": null
    });
    requestPatch(idx, polygon, 0, BACKEND_URL);

    // The following code is for connecting to multiple backends at once
    // for(var i=0; i<ENDPOINTS.length; i++){
    //     //console.debug("Running requestPatch on " + ENDPOINTS[i]["url"]);
    //     currentPatches[idx]["patches"].push({
    //         "srcs": null
    //     });
    //     requestPatch(idx, polygon, i, BACKEND_URL); //TODO: this should be changed if we want to have a web tool that queries different backends
    // }
};

var requestPatch = function(idx, polygon, currentImgIdx, serviceURL){
    var topleft = L.latLng(polygon[0][0], polygon[0][1]);
    var topleftProjected = L.CRS.EPSG3857.project(topleft);
    var bottomright = L.latLng(polygon[2][0], polygon[2][1]);
    var bottomrightProjected = L.CRS.EPSG3857.project(bottomright);

    var request = {
        "type": "runInference",
        "dataset": DATASET,
        "experiment": EXP_NAME,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "spatialReference": {
                "latestWkid": 3857
            }
        },
        "colors": colorList,
    };
    
    $.ajax({
        type: "POST",
        url: serviceURL + "predPatch",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            var resp = data;

            var srcs = [
                "data:image/png;base64," + resp.output_soft,
                "data:image/png;base64," + resp.output_hard,
            ];
            
            var img = $("#exampleImage_"+currentImgIdx);
            img.attr("src", srcs[soft0_hard1]);
            img.attr("data-name", resp.model_name);                    

            if(currentImgIdx == currentPatches[idx]["activeImgIdx"]){
                img.addClass("active");

                if(pred0_naip1 == 0){
                    //var imageLayer = L.imageOverlay(srcs[soft0_hard1], L.polygon(polygon).getBounds()).addTo(map);
                    currentPatches[idx]["imageLayer"].setUrl(srcs[soft0_hard1]);
                }
            }
            currentPatches[idx]["patches"][currentImgIdx]["srcs"] = srcs;
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
        "dataset": DATASET,
        "experiment": EXP_NAME,
        "extent": {
            "xmax": bottomrightProjected.x,
            "xmin": topleftProjected.x,
            "ymax": topleftProjected.y,
            "ymin": bottomrightProjected.y,
            "spatialReference": {
                "latestWkid": 3857
            }
        }
    };

    $.ajax({
        type: "POST",
        url: serviceURL + "getInput",
        data: JSON.stringify(request),
        success: function(data, textStatus, jqXHR){
            var resp = data;
            var naipImg = "data:image/png;base64," + resp.input_naip;
            currentPatches[idx]["naipImg"] = naipImg 
            $("#inputImage").attr("src", naipImg);

            if(pred0_naip1 == 1){
                //var imageLayer = L.imageOverlay(naipImg, L.polygon(polygon).getBounds()).addTo(map);
                currentPatches[idx]["imageLayer"].setUrl(naipImg);
            }
        },
        error: notifyFail,
        dataType: "json",
        contentType: "application/json"
    });
};