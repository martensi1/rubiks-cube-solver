import numpy as np
import cv2


class CubeColor:
    """
    Class representing a color of a Rubik's cube.
    """
    def __init__(self, name, color_range, visualization_color, orient_help_color, face):
        self.name = name
        self.color_range = color_range
        self.visualization_color = visualization_color
        self.orient_help_color = orient_help_color
        self.face = face

    name: str
    color_range: tuple
    visualization_color: tuple
    orient_help_color: tuple


class CubeOrientation:
    """
    Class representing the orientation of the cube in the image
    """
    def __init__(self, origin, cw_next, cw_prev):
        self.origin = origin
        self.cw_next = cw_next
        self.cw_prev = cw_prev

    origin: cv2.Mat
    cw_next: cv2.Mat
    cw_prev: cv2.Mat


class FoundSquare:
    """
    Class representing a square found in the image
    """
    def __init__(self, contour, color):
        self.contour = contour
        self.color = color
        self.number = 0

    contour: np.ndarray
    color: CubeColor
    number: int


class FoundSide:
    """
    Class representing a side of the cube found in the image
    """
    def __init__(self, squares, middle_color):
        self.squares = squares
        self.middle_color = middle_color

    squares: list[FoundSquare]
    middle_color: CubeColor


class ColorList:
    """
    List of all colors of a Rubik's cube
    """
    RED = CubeColor(
        name="red",
        color_range=[(0, 100, 70), (10, 255, 255)],
        visualization_color=(0, 0, 255),
        orient_help_color=(0, 255, 255),
        face="F"
    )
    GREEN = CubeColor(
        name="green",
        color_range=[(50, 100, 70), (100, 255, 255)],
        visualization_color=(0, 255, 0),
        orient_help_color=(0, 255, 255),
        face="R"
    )
    BLUE = CubeColor(
        name="blue",
        color_range=[(100, 100, 70), (130, 255, 255)],
        visualization_color=(255, 0, 0),
        orient_help_color=(0, 255, 255),
        face="L"
    )
    ORANGE = CubeColor(
        name="orange",
        color_range=[(10, 100, 70), (20, 255, 255)],
        visualization_color=(0, 165, 255),
        orient_help_color=(0, 255, 255),
        face="B"
    )
    YELLOW = CubeColor(
        name="yellow",
        color_range=[(20, 100, 70), (40, 255, 255)],
        visualization_color=(0, 255, 255),
        orient_help_color=(0, 165, 255),
        face="U"
    )
    WHITE = CubeColor(
        name="white",
        color_range=[(0, 0, 100), (180, 110, 255)],
        visualization_color=(255, 255, 255),
        orient_help_color=(0, 0, 255),
        face="D"
    )

    ALL = [RED, GREEN, BLUE, ORANGE, YELLOW, WHITE]