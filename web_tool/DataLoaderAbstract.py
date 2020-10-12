import abc

class DataLoader(abc.ABC):

    @property
    @abc.abstractmethod
    def padding(self):
        """A float value that describes the amount of padding (in terms of the class' data source CRS) to apply on input shapes in `get_data_from_extent`.  
        """
        pass

    @abc.abstractmethod
    def __init__(self, padding, **kwargs):
        """A `DataLoader` object should be able to query a source of data by polygons / extents and return the result in a reasonable amount of time. This functionality is abstracted
        as different sources of data will need to be queried using different interfaces. For example, local raster data sources (".tif", ".vrt", etc.) can be simply accessed, while global data
        sources provided by a basemap will need more effort to access.

        Args:
            padding (float): Amount of padding in terms of units of the CRS of the raster source pointed to by `data_fn` to apply during `get_data_from_extent`.
            **kwargs: Key, value pairs created from the contents of this implementation's "dataLayer" key in datasets.json.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_data_from_extent(self, extent):
        """Returns the data from the class' data source corresponding to a *buffered* version of the input extent.
        Buffering is done by `self.padding` number of units (in terms of the source coordinate system).

        Args:
            extent (dict): A geographic extent formatted as a dictionary with the following keys: xmin, xmax, ymin, ymax, crs

        Returns:
            output_raster (InMemoryRaster): A raster cropped to a *buffered* version of the input extent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_from_geometry(self, geometry):
        """Returns the data from the class' raster source corresponding to the input `geometry` without any buffering applied. Note that `geometry` is expected
        to be a GeoJSON polygon (as a dictionary) in the EPSG:4326 coordinate system.

        Args:
            geometry (dict): A polygon in GeoJSON format describing the boundary to crop the input raster to

        Returns:
            output_raster (InMemoryRaster): A raster cropped to the outline of `geometry`
        """
        raise NotImplementedError()