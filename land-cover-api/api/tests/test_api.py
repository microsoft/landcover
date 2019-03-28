import unittest
import base64
from unittest import TestCase
from runserver import app

class LatlonInputJson():
    def __init__(lat, lon, patch_size, latest_wkid, weights=None):
        self.lat = lat
        self.lon = lon
        self.patch_size = patch_size
        self.latest_wkid = latest_wkid
        self.weights = weights

class ExtentInputJson():
    def __init__(xmin, xmax, ymin, ymax, latest_wkid, weights=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.latest_wkid = latest_wkid
        self.weights = weights
    
class APITest(TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.headers ={'Content-Type': 'application/json'}
        
        self.saved_input_tile_img_path = "/tests/input_tile.png"
        self.saved_output_hard_img_path = "/tests/output_hard.png"
        
        self.input_naip_key = "input_naip"
        self.output_hard_key = "output_hard"
           
        self.setup_inputs()

     def setup_inputs(self):       
        input_latlon = LatlonInputJson(47.64432195, -122.136437243892, 3857, 500)
        
        self.input_json_latlon_tile = input_latlon
        self.input_json_latlon_output = input_latlon
        self.input_json_latlon_output.weights = [0.25, 0.25, 0.25, 0.25]

        input_extent = ExtentInputJson(-13596416.0, -13595916.0, 6047634, 6048135.999999998, 3857)
        input_extent.weights = None

        self.input_json_extent_tile = input_extent
        self.input_json_extent_output = input_extent
        self.input_json_extent_output.weights = [0.25, 0.25, 0.25, 0.25]

    def test_health(self):
        with self.app as client:
            resp = client.get('/')

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_data(as_text=True), "Health check OK")

    def test_get_tile(self):
        with self.app as client:
            api_url = '/tile'
            
            resp = client.post(api_url, data=self.input_json_latlon_tile, 
                               headers=self.headers)
            json_data = json.loads(resp.data)

            self.assert_outputs(json_data, self.saved_input_tile_img_path,
                                self.input_naip_key )

    def test_classify(self):
        with self.app as client:
            api_url = '/classify'
                    
            resp = client.post(api_url, data=self.input_json_latlon_output,
                               headers=self.headers)            
            json_data = json.loads(resp.data)
        
            self.assert_outputs(json_data, self.saved_output_hard_img_path,
                                self.output_hard_key)

    def test_get_tile_by_extent(self):
        with self.app as client:
            api_url = 'tile_by_extent'
            
            resp = client.post(api_url, data=self.input_json_extent_tile, 
                               headers=self.headers)
            json_data = json.loads(resp.data)
   
            self.assert_outputs(json_data, self.saved_input_tile_img_path,
                                self.input_naip_key )
            
    def test_classify_by_extent(self):
        with self.app as client:
            api_url = 'classify_by_extent'
            
            resp = client.post(api_url, data=self.input_json_extent_output,
                               headers=self.headers)            
            json_data = json.loads(resp.data)
        
            self.assert_outputs(json_data, self.saved_output_hard_img_path,
                                self.output_hard_key)
          
    def get_base64_img_str(self, image_file_path):
        with open(image_file_path, "rb") as imageFile:
            str = base64.b64encode(imageFile.read())
            return str

    def assert_outputs(self,json_data, saved_img_path, key):
        saved_img = get_base64_img_str(saved_img_path)

        self.assertEqual(resp.status_code, 200)
        self.assert key in reponse_data
        self.assertEqual(json_data[key], saved_img)


if __name__ == '__main__':
    unittest.main()
