var tileLayers = {
    "esri_world_imagery": {
        "location": [[38, -88], 4, "ESRI World Imagery"],
        "tileObject": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 20,
            maxNativeZoom: 17,
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }),
        "shapes": null 
    },
    "esri_world_imagery_naip": {
        "location": [[38, -88], 4, "ESRI World Imagery"],
        "tileObject": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 20,
            maxNativeZoom: 17,
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }),
        "shapes": null 
    },
    "osm": {
        "location": [[38, -88], 4, "OpenStreetMap"],
        "tileObject": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 17,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }),
        "shapes": null 
    },
    "chesapeake": {
        "location": [[38.11437, -75.99980], 10, 'NAIP Imagery - Potential wetlands'],
        "tileObject": L.tileLayer('tiles/chesapeake_test/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 2
        }),
        "shapes": "shapes/chesapeake_test_outline.geojson" 
    },
    "demo_set_1": {
        "location": [[39.40625604822793804, -76.5937627969694006], 13, "User study - Demo area"],
        "tileObject": L.tileLayer('tiles/demo_set_1/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/demo_set_1_boundary.geojson" 
    },
    "user_study_1": {
        "location": [[42.406253302897575, -77.12504090737812135], 13, "User study - Area 1"],
        "tileObject": L.tileLayer('tiles/training_set_1/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/training_set_1_boundary.geojson"
    },
    "user_study_2": {
        "location": [[42.40625552034823897, -76.87503698687157794], 13, "User study - Area 2"],
        "tileObject": L.tileLayer('tiles/training_set_2/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/training_set_2_boundary.geojson" 
    },
    "user_study_3": {
        "location": [[42.46875623949721046, -76.50003291357666058], 13, "User study - Area 3"],
        "tileObject": L.tileLayer('tiles/training_set_3/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/training_set_3_boundary.geojson" 
    },
    "user_study_4": {
        "location": [[43.09375587600650448, -76.18754117285706684], 13, "User study - Area 4"],
        "tileObject": L.tileLayer('tiles/training_set_4/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/training_set_4_boundary.geojson"
    },
    "user_study_5": {
        "location": [[42.448269618302362, -75.110429001207137], 13, "User study area"],
        "tileObject": L.tileLayer('tiles/user_study_5/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": "shapes/user_study_5_outline.geojson"
    },
    "philipsburg_mt": {
        "location": [[46.330963, -113.296773], 13, 'Philipsburg, MT'],
        "tileObject": L.tileLayer('tiles/philipsburg/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 20,
            maxNativeZoom: 18,
            minZoom: 13
        }),
        "shapes": null 
    },
    "yangon": {
        "location": [[16.66177, 96.326427], 10, 'Yangon, Myanmar'],
        "tileObject": L.tileLayer('tiles/yangon/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 17,
            maxNativeZoom: 16,
            minZoom: 10
        }),
        "shapes": "shapes/yangon_grid_shapes.geojson" 
    },
    "aceh": { // Planet data from 3-month mosaics covering the Aceh region of Sumatra
        "location": [[3.68745, 97.47070], 7, "Aceh, Sumatra"],
        "tileObject": L.tileLayer('tiles/leuser/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom: 17,
            maxNativeZoom: 16,
            minZoom: 10
        }),
        "shapes": null 
    },
    "hcmc": {
        "location": [[10.83898610719171, 106.740692498798225], 14, "Thủ Đức District, Hồ Chí Minh City, Vietnam"],
        "tileObject": L.tileLayer('tiles/HCMC/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom:21,
            maxNativeZoom: 18,
            minZoom: 14
        }),
        "shapes": "shapes/hcmc_wards.geojson" 
    },
    "hcmc_sentinel": {
        "location": [[10.83898610719171, 106.740692498798225], 14, "Hồ Chí Minh City, Vietnam / Sentinel"],
        "tileObject": L.tileLayer('tiles/hcmc_sentinel/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom:17,
            maxNativeZoom: 16,
            minZoom: 10
        }),
        "shapes": "shapes/hcmc_sentinel_districts.geojson" 
    },
    "yangon_lidar": {
        "location": [[16.7870, 96.1450], 13, "Yangon, Myanmar"],
        "tileObject": L.tileLayer('tiles/yangon_lidar/{z}/{x}/{y}.png', {
            attribution: 'Georeferenced Image', 
            tms:true,
            maxZoom:21,
            maxNativeZoom: 20,
            minZoom: 10
        }),
        "shapes": "shapes/yangon_wards.geojson" 
    }
};

var interestingLocations = [
    L.marker([47.60, -122.15]).bindPopup('Bellevue, WA'),
    L.marker([39.74, -104.99]).bindPopup('Denver, CO'),
    L.marker([37.53,  -77.44]).bindPopup('Richmond, VA'),
    L.marker([39.74, -104.99]).bindPopup('Denver, CO'),
    L.marker([37.53,  -77.44]).bindPopup('Richmond, VA'),
    L.marker([33.746526, -84.387522]).bindPopup('Atlanta, GA'),
    L.marker([32.774250, -96.796122]).bindPopup('Dallas, TX'),
    L.marker([40.106675, -88.236409]).bindPopup('Champaign, IL'),
    L.marker([38.679485, -75.874667]).bindPopup('Dorchester County, MD'),
    L.marker([34.020618, -118.464412]).bindPopup('Santa Monica, CA'),
    L.marker([37.748517, -122.429771]).bindPopup('San Fransisco, CA'),
    L.marker([38.601951, -98.329227]).bindPopup('Ellsworth County, KS')
];