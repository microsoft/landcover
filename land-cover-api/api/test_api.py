import unittest
import base64
import json
from unittest import TestCase
from runserver import app
from os import getenv
from generate_input_data import get_tile_input_json, Options

class APITest(TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.headers ={'Content-Type': 'application/json'}
        
        self.input_naip_key = "input_naip"
        self.output_hard_key = "output_hard"

        self.api_prefix =  getenv('API_PREFIX')
           

    def test_health(self):
        with self.app as client:
            resp = client.get('/')

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_data(as_text=True), "Health check OK")

    def test_get_tile(self):
        with self.app as client:
            api_url = self.api_prefix + '/tile'

            data = get_tile_input_json(Options.tile.value, random=False)
            resp = client.post(api_url, json=data, headers=self.headers)
            json_data = json.loads(resp.data)

            self.assert_outputs(resp.status_code, json_data, self.input_naip_key )

    def test_classify(self):
        with self.app as client:
            api_url = self.api_prefix + '/classify'

            data = get_tile_input_json(Options.classify.value, random=False)        
            resp = client.post(api_url, json=data,headers=self.headers)            
            json_data = json.loads(resp.data)
        
            self.assert_outputs(resp.status_code, json_data, self.output_hard_key)

    def test_get_tile_by_extent(self):
        with self.app as client:
            api_url = self.api_prefix +'/tile_by_extent'

            data = get_tile_input_json(Options.tilebyExtent.value, random=False)
           
            resp = client.post(api_url, json=data, headers=self.headers)
            json_data = json.loads(resp.data)
            
            self.assert_outputs(resp.status_code, json_data, self.input_naip_key)
            
    def test_classify_by_extent(self):
        with self.app as client:
            api_url = self.api_prefix + '/classify_by_extent'

            data = get_tile_input_json(Options.classifybyExtent.value, random=False)
            resp = client.post(api_url, json=data, headers=self.headers)
            json_data = json.loads(resp.data)
        
            self.assert_outputs(resp.status_code, json_data, self.output_hard_key)
   
    def assert_outputs(self, status_code, json_data, key):
        self.assertEqual(status_code, 200)
        self.assertIn(key, json_data)


if __name__ == '__main__':
    unittest.main()
