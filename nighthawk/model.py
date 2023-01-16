import nighthawk_new as nh


_INPUT_SAMPLE_RATE = 22050    # hertz
_INPUT_DURATION = 1           # seconds


class Model:


    def __init__(self):

        # This is here instead of at the top of this file so that
        # importing this module does not necessarily trigger a
        # TensorFlow import. You have to actually initialize a model
        # to trigger a TensorFlow import.
        import tensorflow as tf

        self._model = tf.saved_model.load(nh.MODEL_DIR_PATH)
        self._input_length = int(round(_INPUT_DURATION * _INPUT_SAMPLE_RATE))
        self._output_taxa = _get_output_taxa()


    @property
    def input_sample_rate(self):

        """Model input sample rate in hertz."""

        return _INPUT_SAMPLE_RATE


    @property
    def input_length(self):

        """Model input length in samples."""

        return self._input_length


    @property
    def output_taxa(self):

        """
        Model output taxa, a tuple of taxon tuples, with taxon (i, j)
        that of elements (i, :, j) of model output.
        """

        return self._output_taxa


    def process(self, samples):

        """
        Apply model to one frame of input samples.

        Parameters
        ----------
        samples : ndarray[float32]
            One frame of input samples in a 1-D NumPy `ndarray`. The frame
            must have length `self.input_length` and a sample rate of
            `self.input_sample_rate`. Its `dtype` must be `float32`.

        Returns
        -------
        list[Tensor]
            List of model output logit tensors. The list comprises four
            2-D tensors, one each for taxonomic order, family, group, and
            species. The first dimension of all of the tensors is one. The
            second dimension is the number of model taxa of the appropriate
            taxonomic rank.
        """

        return self._model(samples)


def _get_output_taxa():
    
    def get_taxa(rank):
        file_path = nh.TAXONOMY_DIR_PATH / f'{rank}_select_v5.txt'
        with open(file_path, 'r') as f:
            return tuple(f.read().split())

    # Return tuple of tuples of taxon names, one tuple per taxonomic rank.
    return tuple(get_taxa(rank) for rank in nh.TAXONOMIC_RANK_PLURALS)
