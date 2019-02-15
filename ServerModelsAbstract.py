
class BackendModel(object):

    @abstractmethod
    def run(self, naip_data, naip_fn, extent, buffer):
        raise NotImplementedError()