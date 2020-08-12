import abc

class ModelSession(abc.ABC):

    @property
    @abc.abstractmethod
    def last_tile(self):
        """This property should be updated by `run()` with the value of the last `tile` tensor that was passed
        when `inference_mode == False`.

        The purpose of keeping track of this data is to provide context for the `row` and `col` indices used
        in `add_sample_point()`. This property does not need to be serialized to/from disk during
        `save_state_to()` and `load_state_from()`.
        """
        pass

    @abc.abstractmethod
    def __init__(self, gpu_id, **kwargs):
        """Responsible for initializing the model and other necessary components from the parameters in the
        models.json files.

        Args:
            gpu_id: An int specifying which GPU to bind to, or None, to specify CPU.
            **kwargs: Key, value pairs created from the contents of this implementation's "model" key in models.json.
                (the model filename should be passed this way)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, tile, inference_mode=False):
        """Responsible for running the model on arbitrarily sized inputs.

        Args:
            tile: A tensor of data of size `(height, width, channels)` that has been cropped from the data source
                currently in use on the front-end. Here, `height` and `width` should be expected to
                vary between calls to `run()`.
            inference_mode: A boolean indicating whether or not to store the `tile` argument in `self.last_tile`.
                This should be `True` when the purpose of calling run is just for executing the model
                (vs. for executing and fine-tuning the model).

        Returns:
            A tensor of size `(height, width, num_classes)` where the last dimension sums to 1
                (e.g. as a result of applying the softmax function to the vector at every spatial location).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def retrain(self, **kwargs):
        """Responsible for updating the parameters of the internal model given the fine-tuning samples
        that have been passed through `add_sample_point()`.
        The mechanism by which this happen is entirely up to the implementation of the class. Some
        implementations may use _all_ previously submitted fine-tuning samples, while other implementations
        may use only the samples submitted since the last call to `retrain()`.

        Returns:
            Dictionary in the format `{"message": str, "success": bool}` describing the results of the retrain.
            The "message" will be displayed as HTML on the front-end, and styled according to "success".
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add_sample_point(self, row, col, class_idx):
        """Responsible for recording fine-tuning samples internally so that they can be used in the next
        call to `retrain()`. Called once for every fine-tuning sample submitted in the front-end interface.

        Args:
            row: The row index into the last `tile` tensor that was passed to `run()`.
                This tensor should be stored in `self.last_tile`.
            col: The column index into the last `tile` tensor that was passed to `run()`.
                This tensor should be stored in `self.last_tile`.
            class_idx: The new class label (0 indexed) that is associated with the given
                `row` and `column` of `self.last_tile`.

        Returns:
            Dictionary in the format `{"message": str, "success": bool}` describing the results of trying to
            add a training sample. The "message" will be displayed as HTML on the front-end, and styled
            according to "success".
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        '''Responsible for resetting the state of the internal model back to the initial configuration
        that it was read "from disk".

        Note: This is not necessarily the original state of the model. If the (ModelSession) class was
        serialized from disk it should be reset to that state.

        Returns:
            Dictionary in the format `{"message": str, "success": bool}` describing the result of
            the reset operation. The "message" will be displayed as HTML on the front-end, and styled
            according to "success".
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def undo(self):
        """Responsible for removing the previously added fine-tuning sample (from `add_sample_point()`)
        or rolling back a model training step - up to the implementation.

        Returns:
            Dictionary in the format `{"message": str, "success": bool}` describing the results of
            the undo operation. The "message" will be displayed as HTML on the front-end, and styled
            according to "success".
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_state_to(self, directory):
        """Responsible for serializing the _current_ state of the class to a directory with the purpose
        of re-hydrating later.

        Args:
            directory: The directory to serialize to. This is guaranteed to exist and
            only contain: "classes.json", "request_replay.p" and "samples.geojson".
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_state_from(self, directory):
        """Responsible for re-hydrating a previously serialized model. After this method is run then the state of
        this object should be such that `run()` can be called immediately after.

        Args:
            directory: The directory to re-hydrate from. This directory should have the output
            from `save_state_to()` in it.
        """
        raise NotImplementedError()