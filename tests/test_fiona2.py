import fiona.transform

geom = {'type': 'Polygon', 'coordinates': [[(440115.0, 4317302.0), (446180.0, 4317302.0), (446180.0, 4309726.0), (440115.0, 4309726.0), (440115.0, 4317302.0)]]}
geom = fiona.transform.transform_geom("EPSG:26918", "EPSG:4326", geom)
print(geom)

# The expected result is approximately `{'type': 'Polygon', 'coordinates': [[(-75.69160116601654, 39.00268487761508), (-75.62156162389587, 39.00307904086472), (-75.62096527634455, 38.9348126572227), (-75.69093763250096, 38.93441944712403), (-75.69160116601654, 39.00268487761508)]]}`
assert geom["coordinates"][0][0] != (440115.0, 4317302.0)
