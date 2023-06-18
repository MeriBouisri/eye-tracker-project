class NestedDict(dict):
    """
    A dictionary implementation that allows access to nested dictionaries using dot notation.
    This implementation also overrides the bracket operator (__getitem__) to allow for indexing.
    
    Python dictionaries are ordered as of Python 3.7.

    Examples
    ----------
    >>> dict = {
                'first_level': {
    ...             'second_level': {
    ...                 'third_level': 'hello world!'
    ...             }
    ...         }
    ...     }

    >>> nested_dict = NestedDict(dict)
    >>> nested_value = nested_dict.first_level.second_level.third_level
    >>> print(nested_value)

    hello world!
    """
    def __getattr__(self, attr):
        if attr not in self:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

        value = self[attr]
        
        if isinstance(value, dict):
            return NestedDict(value)
      
        return value
    
    def __getitem__(self, index):
        if isinstance(index, int):
            value = list(self.values())[index]
        else:
            value = super().__getitem__(index)

        if isinstance(value, dict):
            return NestedDict(value)
        
        return value

        
        

