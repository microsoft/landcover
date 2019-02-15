from abc import ABC, abstractmethod

class BackendModel(ABC):

    @abstractmethod
    def run(self, naip_data, naip_fn, extent, buffer):
        raise NotImplementedError()