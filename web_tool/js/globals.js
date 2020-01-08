var BACKEND_URL = "";
var EXP_NAME = "TEST";
var DATASET = null;
var START_CENTER = null;
var START_ZOOM = null;

var SELECTION_SIZE = 300; // number of units (meters) in EPSG:3857 (the number of _actual_ meters this represents will vary based on latitute)
var CORRECTION_SIZE = 1;

var SESSION_ID = null;

var selectionBox = null;
var currentSelection = null;

var rightMouseDown = false;
var shiftKeyDown = false;
var ctrlKeyDown = false;

var currentPatches = [];
var currentZone = null;
var currentZoneLayerName = null;
var currentBasemapLayerName = null;

var visible = true;
var opacitySlider;

var soft0_hard1 = 1;
var activeImgIdx = 0;

var selectedClassIdx = 0;
var classes = [
    {"name": "Water", "color": "#0000FF", "count": 0},
    {"name": "Tree Canopy", "color": "#008000", "count": 0},
    {"name": "Field", "color": "#80FF80", "count": 0},
    {"name": "Built", "color": "#806060", "count": 0},
];

var retrainCounts = 0
var retrainArgs = {
};

var map = null;
var animating = false;

var numClicks = 0;

var userPointList = []

var heatmap = null;
var undoInProgress = false;


var defaultZoneLineWeights = [
    5,2,0.7,0.3
]
var zoneMapsWeight = {};
var zoneMaps = {};
var defaultZoneStyle = {
    "fillOpacity": 0,
    "color": "#00ff00",
    "weight": 2,
    "opacity": 1
};

var defaultZoneStyle = function(w){
    return  {
        "fillOpacity": 0,
        "color": "#00ff00",
        "weight": 1.5*w,
        "opacity": 1
    }
};

var highlightedZoneStyle = {
    "fillOpacity": 0,
    "color": "#ff0000",
    "weight": 4,
    "opacity": 1
}