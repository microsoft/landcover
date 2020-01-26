from abc import ABC, abstractmethod

class DataLoader(ABC):

    @property
    @abstractmethod
    def padding(self):
        pass

    @property
    @abstractmethod
    def shapes(self):
        pass

    @abstractmethod
    def get_shape_by_extent(self, extent, shape_layer):
        raise NotImplementedError()

    @abstractmethod
    def get_data_from_extent(self, extent):
        raise NotImplementedError()

    @abstractmethod
    def get_area_from_shape_by_extent(self, extent, shape_layer):
        raise NotImplementedError()

    @abstractmethod
    def get_data_from_shape_by_extent(self, extent, shape_layer):
        raise NotImplementedError()

    @abstractmethod
    def get_data_from_shape(self, shape):
        raise NotImplementedError()