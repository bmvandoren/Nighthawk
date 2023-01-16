"""Module defining class `Postprocessor`."""


class Postprocessor:

    """
    Superclass of Nighthawk detector logit postprocessors.

    A `Postprocessor` processes the logits produced by the detector's
    TensorFlow model to create detections.
    """


    def process(self, logits, start_frame_index):

        """
        Gets detections from model output logits.

        This method must be implemented by `Postprocessor` subclasses.

        Parameters
        ----------
        logits : list[ndarray]
            List of model output logit arrays. The list comprises four
            2-D arrays, one each for taxonomic order, family, group,
            and species. All of the arrays have the same first dimension,
            the number of logit frames. The second dimension is the number
            of model taxa of the appropriate taxonomic rank.

        start_frame_index : int
            The index of the first logit frame.

        Returns
        -------
        DataFrame
            Tuple of detections, each a tuple with the following elements:
                0 - frame index
                1 - order name
                2 - order probability
                3 - family name
                4 - family probability
                5 - group code
                6 - group probability
                7 - species code
                8 - species probability
                9 - taxon name
                10 - taxon probability
        """

        raise NotImplementedError()
