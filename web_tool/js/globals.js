var INFERENCE_WINDOW_SIZE = 300; // number of units (meters) in EPSG:3857 (the number of _actual_ meters this represents will vary based on latitute)
var CORRECTION_WINDOW_SIZE = 1; // same as `INFERENCE_WINDOW_SIZE`

var SESSION_ID = getRandomString(10);


var DEFAULT_ZONE_LINE_WEIGHTS = [
    5,2,0.7,0.3
]

var DEFAULT_ZONE_STYLE = function(w){
    return  {
        "fillOpacity": 0,
        "color": "#00ff00",
        "weight": 1.5*w,
        "opacity": 1
    }
};

var HIGHLIGHTED_ZONE_STYLE = {
    "fillOpacity": 0,
    "color": "#ff0000",
    "weight": 4,
    "opacity": 1
};

var DATASETS = (function () {
    var json = null;
    $.ajax({
        'async': false,
        'url': 'datasets.json',
        'dataType': "json",
        'success': function(data){
            json = data;
        }
    });
    return json;
})();

var CLASSES = [
    {"name": "Water", "color": "#0000FF", "count": 0},
    {"name": "Tree Canopy", "color": "#008000", "count": 0},
    {"name": "Field", "color": "#80FF80", "count": 0},
    {"name": "Built", "color": "#806060", "count": 0},
];

//GUI elements
var gSelectionBox = null;
var gCurrentSelection = null;

var gCustomDrawnItems = new L.FeatureGroup();

var gCurrentPatches = [];
var gCurrentZone = null;
var gCurrentZoneLayerName = null;
var gCurrentBasemapLayerName = null;
var gCurrentDataset = null;
var gCurrentCustomPolygon = null; // this holds the current active leaflet `L.Layer` object for the custom polygon that the user can create


var gRightMouseDown = false;
var gShiftKeyDown = false;
var gCtrlKeyDown = false;

var gVisible = true;
var gOpacitySlider;

var gDisplayHard = true;
var gActiveImgIdx = 0;

var gBackendURL = ""; // this will be the base URL of the backend server we make requests to, e.g. "http://msrcalebubuntu1.eastus.cloudapp.azure.com:8080/". Note, a trailing "/" is necessary.

var gSelectedClassIdx = 0;
var gUserPointList = []

var gRetrainCounts = 0
var gRetrainArgs = {};
var gMap = null; // global leaflet `L.Map` object
var gAnimating = false;
var gNumClicks = 0;
var gUndoInProgress = false;


var gZoneMapsWeight = {};

var gBasemaps = {};
var gZonemaps = {};