var addCustomLogoControl = function(){
    //---------------------------------------------------------------------
    // Setup the top left logo box.
    // We format this as a leaflet component in order to directly integrate it with the fullscreen leaflet map.
    // Because this is a leaflet component we can't directly create it in the HTML.
    //----------------------------------------------------------------------
    var logoControl = $("<div class='leaflet-control logo-area'></div>");
    logoControl.append("<span class='logo-text'>Microsoft AI for Earth</span>");
    logoControl.append("<br/>");
    logoControl.append("<span class='logo-text-small'>Version: 0.9</span>");
    logoControl.append("<br/>");
    logoControl.append("<span class='logo-text-small'>Location: "+DATASETS[gCurrentDataset]["metadata"]["displayName"]+"</span>");
    
    $(".leaflet-top.leaflet-left").append(logoControl)

    $("#lblModelInput").html(DATASETS[gCurrentDataset]["metadata"]["locationName"]);
    return logoControl;
};


var addZoomControls = function(){
    //----------------------------------------------------------------------
    // Custom initialization of the map zoom controls (versus adding it in the initial map creation) so that we can position it where we want
    //----------------------------------------------------------------------
    var zoomControl = L.control.zoom({
        position:'topleft'
    })
    zoomControl.addTo(gMap);
    return zoomControl;
};


var addOpacitySlider = function(){
    //----------------------------------------------------------------------
    // Setup leaflet-slider plugin
    //----------------------------------------------------------------------
    var opacitySlider = L.control.slider(
        function(value){
            gMap.getPane('labels').style.opacity = value / 100.0;
        }, {
            position: 'bottomleft',
            id: 'opacitySlider',
            orientation: 'horizontal',
            collapsed: true,
            syncSlider: true,
            min: 0,
            max: 100,
            value: 100,
            logo: "Opacity",
            size: "171px"
        }
    );
    opacitySlider.addTo(gMap);
    return opacitySlider;
};


var addInferenceWindowSizeSlider = function(){
    var inferenceWindowSizeSlider = L.control.slider(
        function(value){
            INFERENCE_WINDOW_SIZE = value;
        }, {
            position: 'bottomleft',
            id: 'inferenceWindowSizeSlider',
            orientation: 'horizontal',
            collapsed: true,
            syncSlider: true,
            min: 7680,
            max: 23040,
            value: INFERENCE_WINDOW_SIZE,
            logo: "Inference Window Size",
            size: "171px"
        }
    );
    inferenceWindowSizeSlider.addTo(gMap);
    return inferenceWindowSizeSlider;
};


var addCorrectionWindowSizeSlider = function(){
    var correctionWindowSizeSlider = L.control.slider(
        function(value){
            CORRECTION_WINDOW_SIZE = value;
        }, {
            position: 'bottomleft',
            id: 'correctionWindowSizeSlider',
            orientation: 'horizontal',
            collapsed: true,
            syncSlider: true,
            min: 1,
            max: 20,
            value: CORRECTION_WINDOW_SIZE,
            logo: "Correction Window Size",
            size: "171px"
        }
    );
    correctionWindowSizeSlider.addTo(gMap);
    return correctionWindowSizeSlider;
};


var addSharpnessToggleSlider = function(){
    //----------------------------------------------------------------------
    // Setup the sharpness slider to control which type of image is shown
    //----------------------------------------------------------------------
    var sharpnessToggleSlider = L.control.slider(
        function(value){
            gDisplayHard = value == 1; 

            for(idx=0; idx<gCurrentPatches.length; idx++){
                var tActiveImgIdx = gCurrentPatches[idx]["activeImgIdx"];
                var srcs = gCurrentPatches[idx]["patches"][tActiveImgIdx]["srcs"];
                if(gDisplayHard){
                    gCurrentPatches[idx]["imageLayer"].setUrl(srcs["hard"]);
                }else{
                    gCurrentPatches[idx]["imageLayer"].setUrl(srcs["soft"]);
                }
            }

            if(gCurrentPatches.length>0){
                var idx = gCurrentPatches.length - 1;
                for(var tActiveImgIdx=0; tActiveImgIdx<gCurrentPatches[idx]["patches"].length; tActiveImgIdx++){
                    var srcs = gCurrentPatches[idx]["patches"][tActiveImgIdx]["srcs"];
                    if(gDisplayHard){
                        $("#exampleImage_"+tActiveImgIdx).attr("src", srcs["hard"]);
                    }else{
                        $("#exampleImage_"+tActiveImgIdx).attr("src", srcs["soft"]);
                    }
                }
            }

        }, {
            position: 'bottomleft',
            id: 'sharpnessToggleSlider',
            orientation: 'horizontal',
            collapsed: true,
            syncSlider: true,
            min: 0,
            max: 1,
            value: 1,
            logo: "Sharpness",
            size: "171px"
        }
    );
    sharpnessToggleSlider.addTo(gMap);
    return sharpnessToggleSlider;
};


var addSideBar = function(){
    //----------------------------------------------------------------------
    // Setup leaflet-sidebar-v2 and open the "#home" tab 
    //----------------------------------------------------------------------
    var sidebar = L.control.sidebar(
        'sidebar', {
            position: 'right'
        }
    )
    sidebar.addTo(gMap);
    sidebar.open("home")
    return sidebar;
};


var addDrawControls = function(){
    //----------------------------------------------------------------------
    // Add the custom drawn items layer to the global map and create the appropirate control item
    //----------------------------------------------------------------------
    L.Util.setOptions(gCustomDrawnItems, {pane: "customPolygons"});
    gMap.addLayer(gCustomDrawnItems);

    var drawControl = new L.Control.Draw({
        edit: {
            featureGroup: gCustomDrawnItems,
            edit: false,
            remove: true
        },
        draw: {
            polyline: false,
            circle: false,
            circlemarker: false,
            rectangle: false,
            marker: false,
        }
    });
    gMap.addControl(drawControl);
    return drawControl;
};


var addBasemapPickerControl = function(basemaps){
    var basemapPickerControl = L.control.layers(
        basemaps, null, {
            collapsed:false,
            position:"bottomleft"
        }
    );
    basemapPickerControl.addTo(gMap);
    return basemapPickerControl;
};


var addZonePickerControl = function(zonemaps){
    var zonePickerControl = L.control.layers(
        zonemaps, null, {
            collapsed:false,
            position:"bottomleft"
        }
    );
    zonePickerControl.addTo(gMap);
    return zonePickerControl;
};


var addUploadControl = function(){
    L.Control.FileLayerLoad.LABEL = '<span class="fa fa-upload"></span>';
    var uploadControl = L.Control.fileLayerLoad({
        fitBounds: true,
        addToMap: false,
        formats: [
            '.geojson'
        ],
        layerOptions: {
            style: DEFAULT_ZONE_STYLE(2),
            onEachFeature: forEachFeatureOnClick,
            pane: "polygons"
        }
    });
    uploadControl.addTo(gMap);
    return uploadControl;
}

var getESRILayer = function(){
    return L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 20,
        maxNativeZoom: 17,
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });
};

var getOSMLayer = function(){
    return L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 20,
        maxNativeZoom: 17,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    });
};