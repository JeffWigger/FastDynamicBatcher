class DynamicBatchIndexError(IndexError):
    """
    Error raised when the number of outputs returned by the infer function of a InferenceModel
    differs from the number of inputs.
    """

    def __init__(self, message):
        """
        The DynamicBatchIndexError is used to propagate the cause of the Errors to users of the DynBatcher.

        :param message: Message to be propagated back
        :type message: str, optional
        """
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"DynamicBatchIndexError: {self.message}"
