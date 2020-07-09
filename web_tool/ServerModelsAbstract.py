from abc import ABC, abstractmethod

class BackendModel(ABC):

    @abstractmethod
    def run(self, imagery, extent, on_tile):
        '''Inputs:
        `imagery` is a (height, width, channels) image that has been cropped from the data source currently in use on the front-end
        `extent` is the extent dictionary that is given by the front-end
        `on_tile` is a flag that specifies whether we are running the model in "download mode" and shouldn't be updating any internal states 

        Outputs:
        `output` should be a (height, width, 4) softmax image (where the last axis sums to 1)
        '''
        raise NotImplementedError()

    @abstractmethod
    def retrain(self):
        '''No inputs, should retrain the base model given the user-provided samples.

        Outputs:
        `success` whether or not the model was successfully retrained
        `message` the error message if retraining was not sucessful, else None
        '''
        raise NotImplementedError()

    @abstractmethod
    def add_sample_point(self, row, col, class_idx):
        '''Takes as input a bounding box and new class label. Should set the area in the previously predicted patch that is covered by the 
        given bounding box to the new class label. No outputs. 
        
        Inputs: 
        `row` row index into the last `imagery` tensor passed to `run()` 
        `col` column index into the last `imagery` tensor passed to `run()`
        `class_idx` new class label (0 indexed)
        '''
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        '''No inputs or outputs, should reset the base model back to the intial configuration that it was in "from disk".
        '''
        raise NotImplementedError()

    @abstractmethod
    def undo(self):
        '''No inputs or outputs, should remove the previously added sample (from add_sample)
        '''
        raise NotImplementedError()