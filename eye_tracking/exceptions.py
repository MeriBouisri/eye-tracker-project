class EyeTrackingException(Exception):
    pass

class NoFaceFound(EyeTrackingException):
    pass

class NoIrisFound(EyeTrackingException):
    pass

class NoFittingEllipseFound(EyeTrackingException):
    pass

class NoCornealReflectionFound(EyeTrackingException):
    pass